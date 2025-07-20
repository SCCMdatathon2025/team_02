import os

import matplotlib.pyplot as plt
import pandas as pd
from google.cloud import bigquery

client = bigquery.Client(project="sepsis-nlp")

GLOBAL_DIR = "data_cache"

os.makedirs(GLOBAL_DIR, exist_ok=True)


def run(query: str, key: str | None = None):
    # If key is provided, check cache first
    if key and os.path.exists(os.path.join(GLOBAL_DIR, f"{key}.csv")):
        return pd.read_csv(os.path.join(GLOBAL_DIR, f"{key}.csv"))

    print(f"Executing query for {key}")

    # Execute query
    results = client.query(query)
    df = results.to_dataframe()

    # Save to cache if key is provided
    if key:
        df.to_csv(os.path.join(GLOBAL_DIR, f"{key}.csv"), index=False)

    return df


def get_18_plus_persons():
    # Query to get distinct person_ids who have never had any visit when under 18
    query = """
WITH patients_with_underage_visits AS (
  SELECT vo.person_id
  FROM `sccm-discovery.rediscover_datathon_2025.visit_occurrence` vo
  JOIN `sccm-discovery.rediscover_datathon_2025.person` p
    ON vo.person_id = p.person_id
  WHERE EXTRACT(YEAR FROM vo.visit_start_datetime) - p.year_of_birth < 18
)
SELECT 
  DISTINCT p.*
FROM `sccm-discovery.rediscover_datathon_2025.visit_occurrence` vo
JOIN `sccm-discovery.rediscover_datathon_2025.person` p
  ON vo.person_id = p.person_id
WHERE vo.person_id NOT IN (SELECT person_id FROM patients_with_underage_visits)
"""

    return run(query, "18_plus_persons")


def get_visit_data(persons: pd.DataFrame):
    # Get unique person IDs and convert to string for SQL IN clause
    person_ids = persons["person_id"].unique()
    person_ids_str = ",".join(map(str, person_ids))

    query = f"""
SELECT * FROM `sccm-discovery.rediscover_datathon_2025.visit_occurrence`
"""

    visits = run(query, "visit_data")
    visits = visits[visits["person_id"].isin(person_ids)]
    return visits


def get_concept_names(concept_ids: list[int], key: str):
    query = f"""
SELECT * FROM `sccm-discovery.rediscover_datathon_2025.concept` WHERE concept_id IN ({",".join([str(i) for i in concept_ids])})
"""
    return run(query, f"{key}_concept_names")


def map_race(persons: pd.DataFrame):
    race_map = {
        "White": "White",
        "Race not stated": "Not Reported",
        "Asian Indian": "Asian",
        "African American": "Black",
        "Other": "Not Reported",
        "No matching concept": "Not Reported",
        "American Indian or Alaska Native": "American Indian or Alaska Native",
        "Other Pacific Islander": "Native Hawaiian or Other Pacific Islander",
        "More than one race": "Mixed",
        "Black or African American": "Black",
        "Asian": "Asian",
        "American Indian": "American Indian or Alaska Native",
        "Native Hawaiian or Other Pacific Islander": "Native Hawaiian or Other Pacific Islander",
        "Unknown racial group": "Not Reported",
    }
    race_concept_ids = persons["race_concept_id"].unique().tolist()
    race_concept_names = get_concept_names(race_concept_ids, "race")
    persons["race_name"] = persons["race_concept_id"].map(
        race_concept_names.set_index("concept_id")["concept_name"]
    )
    persons["race_name"] = persons["race_name"].fillna("Not Reported")
    persons["race_name"] = persons["race_name"].map(race_map)
    return persons


def get_mortality_data(persons: pd.DataFrame):
    # Get death data for the persons we're interested in
    person_ids = persons["person_id"].unique()
    person_ids_str = ",".join(map(str, person_ids))

    query = f"""
    WITH ranked_deaths AS (
        SELECT 
            person_id,
            death_date,
            death_datetime,
            ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY death_date, death_datetime) as rn
        FROM `sccm-discovery.rediscover_datathon_2025.death`
        WHERE person_id IN ({person_ids_str})
    )
    SELECT 
        person_id,
        death_date,
        death_datetime
    FROM ranked_deaths
    WHERE rn = 1
    """

    deaths = run(query, "mortality_data")

    return deaths


def calculate_num_visits(persons, visits):
    # Do a groupby on person_id and count the number of visits
    visits_per_person = visits.groupby("person_id").size()
    persons["num_visits"] = persons["person_id"].map(visits_per_person)


def map_gender(persons: pd.DataFrame):
    gender_concept_ids = persons["gender_concept_id"].unique().tolist()
    gender_concept_names = get_concept_names(gender_concept_ids, "gender")
    persons["gender_name"] = persons["gender_concept_id"].map(
        gender_concept_names.set_index("concept_id")["concept_name"]
    )
    return persons


def main():
    persons = get_18_plus_persons()
    visits = get_visit_data(persons)
    map_race(persons)
    map_gender(persons)

    # Visits has a person_id, map the race_name to the person_id
    visits["race_name"] = visits["person_id"].map(
        persons.set_index("person_id")["race_name"]
    )

    # Convert visit_start_datetime to pandas datetime for sorting
    visits["visit_start_datetime"] = pd.to_datetime(
        visits["visit_start_datetime"], format="mixed"
    )

    # Convert datetime columns to pandas datetime (handle mixed date/datetime formats)
    visits["visit_end_datetime"] = pd.to_datetime(
        visits["visit_end_datetime"], format="mixed"
    )

    # Sort visits by person_id and visit_start_datetime to identify first visits
    visits = visits.sort_values(["person_id", "visit_start_datetime"])

    # Add first_visit column - True for first visit of each person, False otherwise
    visits["first_visit"] = ~visits.duplicated(subset="person_id", keep="first")

    # Now, also remove visits that are after today
    today = pd.Timestamp.now()
    invalid_visits = visits["visit_end_datetime"] > today
    num_removed = invalid_visits.sum()
    persons_removed = visits[invalid_visits]["person_id"].unique()
    print(f"Removed {num_removed} ({len(persons_removed)}) visits that are after today")
    visits = visits[~invalid_visits]

    # Load mortality data
    deaths = get_mortality_data(persons)

    # Merge death data with visits
    visits = visits.merge(
        deaths[["person_id", "death_date", "death_datetime"]],
        on="person_id",
        how="left",
    )

    visits["death_datetime"] = pd.to_datetime(visits["death_datetime"], format="mixed")
    visits["death_date"] = pd.to_datetime(visits["death_date"], format="mixed")

    # Use death_datetime if available, otherwise use death_date
    visits["effective_death_datetime"] = visits["death_datetime"].fillna(
        visits["death_date"]
    )

    # Calculate days difference (death - discharge)
    visits["days_death_to_discharge"] = (
        visits["effective_death_datetime"] - visits["visit_end_datetime"]
    ).dt.total_seconds() / (24 * 3600)

    # Remove visits where person died 2+ days before discharge
    invalid_visits = visits["days_death_to_discharge"] <= -2
    num_removed = invalid_visits.sum()

    persons_removed = visits[invalid_visits]["person_id"].unique()

    print(
        f"Removed {num_removed} ({len(persons_removed)}) visits where person died 2+ days before discharge"
    )

    visits = visits[~invalid_visits]

    calculate_num_visits(persons, visits)

    # Filter people who are coming > 50 times
    invalid_visits = persons["num_visits"] > 50
    num_removed = invalid_visits.sum()
    persons_removed = persons[invalid_visits]["person_id"].unique()
    print(
        f"Removed {num_removed} ({len(persons_removed)}) visits where person came > 50 times"
    )

    persons = persons[~invalid_visits]
    visits = visits[visits["person_id"].isin(persons["person_id"])]

    calculate_num_visits(persons, visits)

    # Add mortality labels
    visits["died_in_hospital"] = (
        (visits["days_death_to_discharge"] >= -1)
        & (visits["days_death_to_discharge"] <= 0)
    ).fillna(False)

    visits["died_in_30_days"] = (
        (visits["days_death_to_discharge"] > 0)
        & (visits["days_death_to_discharge"] <= 30)
    ).fillna(False)

    # Clean up temporary columns if desired
    visits = visits.drop(
        [
            "death_date",
            "death_datetime",
            "effective_death_datetime",
            "days_death_to_discharge",
        ],
        axis=1,
    )

    # # pllot this
    # plt.figure(figsize=(12, 8))

    # plt.hist(
    #     persons["num_visits"],
    #     bins=range(0, int(persons["num_visits"].max()) + 2),
    #     alpha=0.7,
    #     edgecolor="black",
    #     color="skyblue",
    # )
    # plt.xlabel("Number of Visits per Patient")
    # plt.ylabel("Number of Patients")
    # plt.title("Distribution of Visits per Patient")
    # plt.grid(True, alpha=0.3)
    # plt.show()
    # exit()

    # Now, finally, only keep visits that are the last recorded for each person
    visits = visits.sort_values(["person_id", "visit_start_datetime"])
    visits = visits.drop_duplicates(subset="person_id", keep="last")

    # Only keep people which have at least 1 visit
    persons = persons[persons["num_visits"] > 0]
    visits = visits[visits["person_id"].isin(persons["person_id"])]

    # Only keep visits which have a person_id
    visits = visits[visits["person_id"].notna()]

    calculate_num_visits(persons, visits)

    print("--------------------------------")
    print(f"Total visits: {len(visits)}")
    print(f"Total people: {len(persons)}")
    print("--------------------------------")

    for race in visits["race_name"].unique():
        race_df = visits[visits["race_name"] == race]
        unique_people = race_df["person_id"].nunique()
        print(f"\n{race} (Total visits: {len(race_df)}, People: {unique_people})")

        # Deaths in first visit
        first_visit_deaths = race_df[
            (race_df["first_visit"] == True)
            & (
                (race_df["died_in_hospital"] == True)
                | (race_df["died_in_30_days"] == True)
            )
        ]
        first_visit_death_count = first_visit_deaths["person_id"].nunique()
        first_visit_death_rate = (first_visit_death_count / unique_people) * 100
        print(
            f"  Died in first visit: {first_visit_death_count} ({first_visit_death_rate:.1f}%)"
        )

        # Readmission statistics
        visits_per_person = race_df.groupby("person_id").size()
        readmissions_per_person = (
            visits_per_person - 1
        )  # Subtract 1 since first visit isn't a readmission

        people_with_readmissions = (readmissions_per_person > 0).sum()
        readmission_rate = (people_with_readmissions / unique_people) * 100

        mean_readmissions = readmissions_per_person.mean()
        median_readmissions = readmissions_per_person.median()
        max_readmissions = readmissions_per_person.max()

        print(f"  Readmission statistics:")
        print(
            f"    People with readmissions: {people_with_readmissions} ({readmission_rate:.1f}%)"
        )
        print(f"    Mean readmissions per person: {mean_readmissions:.1f}")
        print(f"    Median readmissions per person: {median_readmissions:.1f}")
        print(f"    Max readmissions: {max_readmissions}")

        # Overall mortality (per people, not visits)
        # People who died in hospital (any visit)
        people_died_hospital = race_df[race_df["died_in_hospital"] == True][
            "person_id"
        ].nunique()
        hospital_mortality_rate = (people_died_hospital / unique_people) * 100

        # People who died within 30 days (any visit)
        people_died_30days = race_df[race_df["died_in_30_days"] == True][
            "person_id"
        ].nunique()
        day30_mortality_rate = (people_died_30days / unique_people) * 100

        # Total people who died (either in hospital or within 30 days)
        people_died_total = race_df[
            (race_df["died_in_hospital"] == True) | (race_df["died_in_30_days"] == True)
        ]["person_id"].nunique()
        total_mortality_rate = (people_died_total / unique_people) * 100

        print(f"  Mortality (people-based):")
        print(
            f"    Died in hospital: {people_died_hospital} ({hospital_mortality_rate:.1f}%)"
        )
        print(
            f"    Died within 30 days: {people_died_30days} ({day30_mortality_rate:.1f}%)"
        )
        print(f"    Total mortality: {people_died_total} ({total_mortality_rate:.1f}%)")

    # Do a join on persons and visits on person_id
    persons = persons.merge(
        visits,
        on=["person_id", "race_name", "provider_id", "care_site_id"],
        how="inner",
    )

    # Now write this to bigquery
    persons.to_gbq(
        "sepsis-nlp.team_2.cohort",
        project_id="sepsis-nlp",
        if_exists="replace",
    )

    print(len(persons))
    print(persons.columns)
    print(persons.head())
    exit()


if __name__ == "__main__":
    main()
