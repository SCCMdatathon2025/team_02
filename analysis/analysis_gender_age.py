import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from google.cloud import bigquery
from tqdm import tqdm

from create_cohort import GLOBAL_DIR, client, get_concept_names, run


def get_cohort():
    query = """
    SELECT * FROM `sepsis-nlp.team_2.cohort`
    """
    return run(query, "cohort")


def get_condition_occurrence_batch(batch_df: pd.DataFrame, batch_idx: int):
    """Process a single batch of person data"""
    person_ids = batch_df["person_id"].unique()
    person_ids_str = ",".join(map(str, person_ids))

    visit_ids = batch_df["visit_occurrence_id"].unique()
    visit_ids_str = ",".join(map(str, visit_ids))

    query = f"""
    SELECT * FROM `sccm-discovery.rediscover_datathon_2025.condition_occurrence` 
    WHERE visit_occurrence_id IN ({visit_ids_str}) AND person_id IN ({person_ids_str})
    """

    # Execute query without caching via run function
    results = client.query(query)
    df = results.to_dataframe()

    return df, batch_idx


def get_condition_occurrence(
    persons: pd.DataFrame, batch_size: int = 1000, max_workers: int = 4
):
    """Get condition occurrence data in batches with parallelization and caching"""

    cache_file = os.path.join(GLOBAL_DIR, "condition_occurrence_batched.csv")

    # Check if cached result exists
    if os.path.exists(cache_file):
        print("Loading condition occurrence data from cache...")
        return pd.read_csv(cache_file)

    print(
        f"Processing {len(persons)} persons in batches of {batch_size} with {max_workers} workers..."
    )

    # Split persons into batches
    batches = [persons[i : i + batch_size] for i in range(0, len(persons), batch_size)]

    all_results = []

    # Process batches in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batch jobs
        future_to_batch = {
            executor.submit(get_condition_occurrence_batch, batch, idx): idx
            for idx, batch in enumerate(batches)
        }

        # Process completed batches with progress bar
        with tqdm(total=len(batches), desc="Processing batches") as pbar:
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_result, _ = future.result()
                    all_results.append(batch_result)
                    pbar.update(1)
                except Exception as exc:
                    print(f"Batch {batch_idx} generated an exception: {exc}")
                    pbar.update(1)

    # Combine all results
    if all_results:
        final_result = pd.concat(all_results, ignore_index=True)
    else:
        final_result = pd.DataFrame()

    # Cache the result
    final_result.to_csv(cache_file, index=False)
    print(f"Cached {len(final_result)} condition occurrence records")

    return final_result


def main(name: str, ids: list[int]):
    cohort = get_cohort()

    race_name_map = {
        "White": "White",
        "Black": "Black",
        "Asian": "Asian",
        "Not Reported": "Not Reported",
    }

    for race in cohort["race_name"].unique():
        if race not in race_name_map:
            race_name_map[race] = "Other"

    cohort["race_name"] = cohort["race_name"].map(race_name_map)

    condition_occurrence = get_condition_occurrence(persons=cohort)

    # Cast to int
    condition_occurrence["condition_status_concept_id"] = (
        condition_occurrence["condition_status_concept_id"].fillna(0).astype("int64")
    )

    condition_status_concept_id_map = {32890: 32890, 32901: 32890, 32907: 32890}

    # Use map with fillna to handle unmapped values
    condition_occurrence["condition_status_concept_id"] = (
        condition_occurrence["condition_status_concept_id"]
        .map(condition_status_concept_id_map)
        .fillna(condition_occurrence["condition_status_concept_id"])
    )

    # Add age calculation and age groups
    cohort["visit_year"] = pd.to_datetime(cohort["visit_start_datetime"]).dt.year
    cohort["age_at_visit"] = cohort["visit_year"] - cohort["year_of_birth"]
    cohort["age_group"] = cohort["age_at_visit"].apply(
        lambda x: "65+" if x >= 65 else "<65"
    )

    print(f"Age distribution:")
    print(f"Mean age: {cohort['age_at_visit'].mean():.1f}")
    print(f"Median age: {cohort['age_at_visit'].median():.1f}")
    print(f"Age range: {cohort['age_at_visit'].min()}-{cohort['age_at_visit'].max()}")
    print(f"Age group distribution:")
    print(cohort["age_group"].value_counts())

    query = f"""
/*───────────────── 1. Cohort (already has demo fields) ─────────────────*/
WITH cohort AS (
  SELECT
    person_id,
    visit_occurrence_id,
    race_concept_id,
    race_name,
    gender_concept_id,
    gender_name,
    year_of_birth,
    visit_start_datetime,
    EXTRACT(YEAR FROM visit_start_datetime) - year_of_birth AS age_at_visit,
    CASE 
      WHEN EXTRACT(YEAR FROM visit_start_datetime) - year_of_birth >= 65 THEN '65+'
      ELSE '<65'
    END AS age_group
  FROM `sccm-discovery.rediscover_datathon_2025.cohort_team_2`
),

/*───────────────── 2. Drug-related visits for the target concept(s) ────*/
visits AS (
  SELECT DISTINCT
         i.person_id,
         i.visit_occurrence_id
  FROM  (SELECT * FROM `sccm-discovery.rediscover_datathon_2025.concept`
         WHERE concept_id IN ({','.join(map(str, ids))}))                            a
  JOIN  `sccm-discovery.rediscover_datathon_2025.concept_relationship` b
         ON a.concept_id = b.concept_id_1
        AND b.relationship_id = 'Subsumes'
  JOIN  `sccm-discovery.rediscover_datathon_2025.concept`              c
         ON c.concept_id = b.concept_id_2
  JOIN  `sccm-discovery.rediscover_datathon_2025.concept_relationship` d
         ON d.concept_id_1 = c.concept_id
        AND d.relationship_id = 'SNOMED - RxNorm eq'
  JOIN  `sccm-discovery.rediscover_datathon_2025.concept`              e
         ON e.concept_id = d.concept_id_2
  JOIN  `sccm-discovery.rediscover_datathon_2025.concept_relationship` f
         ON e.concept_id = f.concept_id_1
  JOIN  (SELECT * FROM `sccm-discovery.rediscover_datathon_2025.concept`
         WHERE standard_concept = 'S')                               g
         ON f.concept_id_2 = g.concept_id
  JOIN  `sccm-discovery.rediscover_datathon_2025.drug_exposure`         h
         ON g.concept_id = h.drug_concept_id
  JOIN  `sccm-discovery.rediscover_datathon_2025.visit_occurrence`      i
         ON h.visit_occurrence_id = i.visit_occurrence_id
),

/*───────────────── 3. Keep only cohort rows that have those drug visits ─*/
cohort_with_drug AS (
  SELECT DISTINCT c.*
  FROM visits v
  JOIN cohort c
    ON v.person_id           = c.person_id
   AND v.visit_occurrence_id = c.visit_occurrence_id
)

/*───────────────── 4. Race × Gender × Age counts ────*/
SELECT
    race_name,
    gender_concept_id,
    gender_name,
    age_group,
    COUNT(DISTINCT person_id) AS cnt      -- change to COUNT(*) for visit-level
FROM cohort_with_drug
GROUP BY
    race_name,
    gender_concept_id, gender_name, age_group
ORDER BY cnt DESC;
"""

    # Compute number of people in cohort by race, gender, and age
    base_cohort_counts = (
        cohort.groupby(["race_name", "gender_name", "age_group"])
        .size()
        .reset_index(name="base_count")
    )

    # Run the query to get drug-related cohort counts
    results = run(query, name)

    # Apply the same race mapping to results that was applied to cohort
    print("Original race names in results:", results["race_name"].unique())
    results["race_name"] = results["race_name"].map(race_name_map).fillna("Other")
    print("Mapped race names in results:", results["race_name"].unique())
    print("Race names in base_cohort_counts:", base_cohort_counts["race_name"].unique())

    # Merge the results to compute ratios
    comparison = results.merge(
        base_cohort_counts, on=["race_name", "gender_name", "age_group"], how="left"
    )

    # Fill missing values with 0 (no one in that group got drugs)
    comparison["cnt"] = comparison["cnt"].fillna(0).astype(int)
    comparison["base_count"] = comparison["base_count"].fillna(0).astype(int)

    # print("\nComparison - Drug cohort representation vs Base cohort:")
    # print(
    #     comparison[
    #         ["race_name", "gender_name", "age_group", "base_count", "cnt"]
    #     ].sort_values("cnt", ascending=False)
    # )

    # Calculate percentages and add statistical significance testing
    comparison["percentage"] = (comparison["cnt"] / comparison["base_count"]) * 100

    # Import statistical testing libraries
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from scipy.stats import chi2_contingency, fisher_exact

    # print("\nPercentage of people who received drugs by race, gender, and age:")
    # print(
    #     comparison[
    #         ["race_name", "gender_name", "age_group", "base_count", "cnt", "percentage"]
    #     ].sort_values("percentage", ascending=False)
    # )
    # STATISTICAL SIGNIFICANCE TESTING WITH AGE STRATIFICATION
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTING - INCLUDING AGE STRATIFICATION")
    print("=" * 80)

    # 1. Overall gender difference test
    gender_summary = (
        comparison.groupby("gender_name")
        .agg({"base_count": "sum", "cnt": "sum"})
        .reset_index()
    )
    gender_summary["percentage"] = (
        gender_summary["cnt"] / gender_summary["base_count"]
    ) * 100

    print("\nOVERALL GENDER SUMMARY:")
    print(gender_summary)

    # Chi-square test for overall gender difference
    male_data = gender_summary[gender_summary["gender_name"] == "MALE"]
    female_data = gender_summary[gender_summary["gender_name"] == "FEMALE"]

    contingency_overall = [
        [
            male_data["cnt"].iloc[0],
            male_data["base_count"].iloc[0] - male_data["cnt"].iloc[0],
        ],
        [
            female_data["cnt"].iloc[0],
            female_data["base_count"].iloc[0] - female_data["cnt"].iloc[0],
        ],
    ]

    chi2_overall, p_overall, dof, expected = chi2_contingency(contingency_overall)

    print(f"\nOVERALL GENDER DIFFERENCE:")
    print(f"Male drug rate: {male_data['percentage'].iloc[0]:.2f}%")
    print(f"Female drug rate: {female_data['percentage'].iloc[0]:.2f}%")
    print(f"Chi-square statistic: {chi2_overall:.4f}")
    print(f"P-value: {p_overall:.6f}")
    print(f"Statistically significant: {'YES' if p_overall < 0.05 else 'NO'}")

    # 2. Overall age difference test
    age_summary = (
        comparison.groupby("age_group")
        .agg({"base_count": "sum", "cnt": "sum"})
        .reset_index()
    )
    age_summary["percentage"] = (age_summary["cnt"] / age_summary["base_count"]) * 100

    print(f"\nOVERALL AGE GROUP SUMMARY:")
    print(age_summary)

    # Chi-square test for overall age difference
    young_data = age_summary[age_summary["age_group"] == "<65"]
    old_data = age_summary[age_summary["age_group"] == "65+"]

    if not young_data.empty and not old_data.empty:
        contingency_age = [
            [
                young_data["cnt"].iloc[0],
                young_data["base_count"].iloc[0] - young_data["cnt"].iloc[0],
            ],
            [
                old_data["cnt"].iloc[0],
                old_data["base_count"].iloc[0] - old_data["cnt"].iloc[0],
            ],
        ]

        chi2_age, p_age, dof, expected = chi2_contingency(contingency_age)

        print(f"\nOVERALL AGE GROUP DIFFERENCE:")
        print(f"<65 drug rate: {young_data['percentage'].iloc[0]:.2f}%")
        print(f"65+ drug rate: {old_data['percentage'].iloc[0]:.2f}%")
        print(f"Chi-square statistic: {chi2_age:.4f}")
        print(f"P-value: {p_age:.6f}")
        print(f"Statistically significant: {'YES' if p_age < 0.05 else 'NO'}")

    # 3. Gender differences within each age group
    print(f"\n" + "-" * 60)
    print("GENDER DIFFERENCES WITHIN EACH AGE GROUP:")
    print("-" * 60)

    age_gender_results = []

    for age_group in comparison["age_group"].unique():
        age_data = comparison[comparison["age_group"] == age_group]

        # Aggregate by gender within this age group
        gender_in_age = (
            age_data.groupby("gender_name")
            .agg({"base_count": "sum", "cnt": "sum"})
            .reset_index()
        )
        gender_in_age["percentage"] = (
            gender_in_age["cnt"] / gender_in_age["base_count"]
        ) * 100

        male_age = gender_in_age[gender_in_age["gender_name"] == "MALE"]
        female_age = gender_in_age[gender_in_age["gender_name"] == "FEMALE"]

        if not male_age.empty and not female_age.empty:
            male_got = male_age["cnt"].iloc[0]
            male_total = male_age["base_count"].iloc[0]
            male_rate = male_age["percentage"].iloc[0]

            female_got = female_age["cnt"].iloc[0]
            female_total = female_age["base_count"].iloc[0]
            female_rate = female_age["percentage"].iloc[0]

            # Create contingency table for this age group
            contingency_age_gender = [
                [male_got, male_total - male_got],
                [female_got, female_total - female_got],
            ]

            # Use appropriate test
            if min(male_total, female_total) < 50:
                oddsratio, p_value = fisher_exact(contingency_age_gender)
                test_used = "Fisher's exact"
            else:
                chi2, p_value, dof, expected = chi2_contingency(contingency_age_gender)
                test_used = "Chi-square"

            diff = male_rate - female_rate

            age_gender_results.append(
                {
                    "age_group": age_group,
                    "male_rate": male_rate,
                    "female_rate": female_rate,
                    "difference": diff,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "test_used": test_used,
                }
            )

            print(f"\nAge Group {age_group}:")
            print(f"  Male rate: {male_rate:.2f}% ({male_got}/{male_total})")
            print(f"  Female rate: {female_rate:.2f}% ({female_got}/{female_total})")
            print(f"  Difference (M-F): {diff:+.2f} percentage points")
            print(f"  {test_used} p-value: {p_value:.6f}")
            print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")

    # 4. Age differences within each gender group
    print(f"\n" + "-" * 60)
    print("AGE DIFFERENCES WITHIN EACH GENDER GROUP:")
    print("-" * 60)

    gender_age_results = []

    for gender in comparison["gender_name"].unique():
        gender_data = comparison[comparison["gender_name"] == gender]

        # Aggregate by age within this gender
        age_in_gender = (
            gender_data.groupby("age_group")
            .agg({"base_count": "sum", "cnt": "sum"})
            .reset_index()
        )
        age_in_gender["percentage"] = (
            age_in_gender["cnt"] / age_in_gender["base_count"]
        ) * 100

        young_gender = age_in_gender[age_in_gender["age_group"] == "<65"]
        old_gender = age_in_gender[age_in_gender["age_group"] == "65+"]

        if not young_gender.empty and not old_gender.empty:
            young_got = young_gender["cnt"].iloc[0]
            young_total = young_gender["base_count"].iloc[0]
            young_rate = young_gender["percentage"].iloc[0]

            old_got = old_gender["cnt"].iloc[0]
            old_total = old_gender["base_count"].iloc[0]
            old_rate = old_gender["percentage"].iloc[0]

            # Create contingency table for this gender
            contingency_gender_age = [
                [young_got, young_total - young_got],
                [old_got, old_total - old_got],
            ]

            # Use appropriate test
            if min(young_total, old_total) < 50:
                oddsratio, p_value = fisher_exact(contingency_gender_age)
                test_used = "Fisher's exact"
            else:
                chi2, p_value, dof, expected = chi2_contingency(contingency_gender_age)
                test_used = "Chi-square"

            diff = old_rate - young_rate

            gender_age_results.append(
                {
                    "gender": gender,
                    "young_rate": young_rate,
                    "old_rate": old_rate,
                    "difference": diff,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "test_used": test_used,
                }
            )

            print(f"\n{gender}:")
            print(f"  <65 rate: {young_rate:.2f}% ({young_got}/{young_total})")
            print(f"  65+ rate: {old_rate:.2f}% ({old_got}/{old_total})")
            print(f"  Difference (65+ minus <65): {diff:+.2f} percentage points")
            print(f"  {test_used} p-value: {p_value:.6f}")
            print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")

    # 5. Gender differences within each race (as before)
    print(f"\n" + "-" * 60)
    print("GENDER DIFFERENCES WITHIN EACH RACE (ACROSS ALL AGES):")
    print("-" * 60)

    race_results = []

    for race in comparison["race_name"].unique():
        race_data = comparison[comparison["race_name"] == race]

        # Aggregate by gender within this race (across age groups)
        gender_in_race = (
            race_data.groupby("gender_name")
            .agg({"base_count": "sum", "cnt": "sum"})
            .reset_index()
        )
        gender_in_race["percentage"] = (
            gender_in_race["cnt"] / gender_in_race["base_count"]
        ) * 100

        male_race = gender_in_race[gender_in_race["gender_name"] == "MALE"]
        female_race = gender_in_race[gender_in_race["gender_name"] == "FEMALE"]

        if not male_race.empty and not female_race.empty:
            male_got = int(male_race["cnt"].iloc[0])
            male_total = int(male_race["base_count"].iloc[0])
            male_rate = male_race["percentage"].iloc[0]

            female_got = int(female_race["cnt"].iloc[0])
            female_total = int(female_race["base_count"].iloc[0])
            female_rate = female_race["percentage"].iloc[0]

            # Validate data integrity and report issues
            if male_got > male_total:
                print(
                    f"ERROR in {race}: Male got drugs ({male_got}) > male total ({male_total})"
                )
                male_got = male_total
            if female_got > female_total:
                print(
                    f"ERROR in {race}: Female got drugs ({female_got}) > female total ({female_total})"
                )
                female_got = female_total

            male_not_got = male_total - male_got
            female_not_got = female_total - female_got

            if male_not_got < 0:
                print(
                    f"ERROR in {race}: Male not_got would be negative ({male_not_got}). Data: got={male_got}, total={male_total}"
                )
                male_not_got = 0
            if female_not_got < 0:
                print(
                    f"ERROR in {race}: Female not_got would be negative ({female_not_got}). Data: got={female_got}, total={female_total}"
                )
                female_not_got = 0

            # Create contingency table for this race
            contingency_race = [
                [male_got, male_not_got],
                [female_got, female_not_got],
            ]

            # Use appropriate test
            if min(male_total, female_total) < 50:
                oddsratio, p_value = fisher_exact(contingency_race)
                test_used = "Fisher's exact"
            else:
                chi2, p_value, dof, expected = chi2_contingency(contingency_race)
                test_used = "Chi-square"

            diff = male_rate - female_rate

            race_results.append(
                {
                    "race": race,
                    "male_rate": male_rate,
                    "female_rate": female_rate,
                    "difference": diff,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "test_used": test_used,
                }
            )

            print(f"\n{race}:")
            print(f"  Male rate: {male_rate:.2f}% ({male_got}/{male_total})")
            print(f"  Female rate: {female_rate:.2f}% ({female_got}/{female_total})")
            print(f"  Difference (M-F): {diff:+.2f} percentage points")
            print(f"  {test_used} p-value: {p_value:.6f}")
            print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")

    # 6. Test if gender differences vary significantly between races
    print(f"\n" + "-" * 60)
    print("TESTING IF GENDER DIFFERENCES VARY BETWEEN RACES:")
    print("-" * 60)

    # Create a comprehensive contingency table for interaction test
    race_gender_table = comparison.pivot_table(
        index="race_name",
        columns="gender_name",
        values=["cnt", "base_count"],
        aggfunc="sum",
        fill_value=0,
    )

    # Test for interaction between race and gender
    # Create 3D contingency table: Race x Gender x (Got drug / Didn't get drug)
    races = sorted(comparison["race_name"].unique())

    interaction_data = []
    for race in races:
        race_data = comparison[comparison["race_name"] == race]
        male_data = race_data[race_data["gender_name"] == "MALE"]
        female_data = race_data[race_data["gender_name"] == "FEMALE"]

        if not male_data.empty and not female_data.empty:
            male_got = int(male_data["cnt"].sum())
            male_total = int(male_data["base_count"].sum())
            female_got = int(female_data["cnt"].sum())
            female_total = int(female_data["base_count"].sum())

            # Validate data integrity and report issues
            if male_got > male_total:
                print(
                    f"ERROR in {race} interaction: Male got drugs ({male_got}) > male total ({male_total})"
                )
                male_got = male_total
            if female_got > female_total:
                print(
                    f"ERROR in {race} interaction: Female got drugs ({female_got}) > female total ({female_total})"
                )
                female_got = female_total

            # Skip if insufficient data
            if male_total == 0 or female_total == 0:
                print(
                    f"WARNING in {race} interaction: Zero totals - male_total={male_total}, female_total={female_total}"
                )
                continue

            male_rate = (male_got / male_total * 100) if male_total > 0 else 0
            female_rate = (female_got / female_total * 100) if female_total > 0 else 0
            gender_diff = male_rate - female_rate

            interaction_data.append(
                {
                    "race": race,
                    "male_got": male_got,
                    "male_total": male_total,
                    "male_rate": male_rate,
                    "female_got": female_got,
                    "female_total": female_total,
                    "female_rate": female_rate,
                    "gender_diff": gender_diff,
                }
            )

    # Display gender differences by race
    print("\nGender differences by race:")
    for data in interaction_data:
        print(
            f"  {data['race']:<20}: M={data['male_rate']:5.1f}% F={data['female_rate']:5.1f}% Diff={data['gender_diff']:+5.1f}pp"
        )

    # Test for homogeneity of gender differences across races
    # Using Breslow-Day test approach or chi-square test of homogeneity
    if len(interaction_data) >= 2:
        # Create contingency table for each race: [male_got, male_not_got, female_got, female_not_got]
        contingency_by_race = []
        for data in interaction_data:
            male_not_got = data["male_total"] - data["male_got"]
            female_not_got = data["female_total"] - data["female_got"]

            if male_not_got < 0:
                print(
                    f"ERROR in {data['race']} contingency: Male not_got negative ({male_not_got})"
                )
                male_not_got = 0
            if female_not_got < 0:
                print(
                    f"ERROR in {data['race']} contingency: Female not_got negative ({female_not_got})"
                )
                female_not_got = 0

            race_table = [
                [data["male_got"], male_not_got],
                [data["female_got"], female_not_got],
            ]
            contingency_by_race.append(race_table)

        # Mantel-Haenszel test for homogeneity of odds ratios
        from scipy.stats import chi2

        # Calculate odds ratios for each race
        odds_ratios = []
        for data in interaction_data:
            if data["male_total"] > 0 and data["female_total"] > 0:
                male_odds = data["male_got"] / max(
                    1, data["male_total"] - data["male_got"]
                )
                female_odds = data["female_got"] / max(
                    1, data["female_total"] - data["female_got"]
                )
                if female_odds > 0:
                    or_value = male_odds / female_odds
                    odds_ratios.append(or_value)
                    print(f"  {data['race']:<20}: OR = {or_value:.3f}")

        # Test if the gender effect varies significantly across races
        # Create a combined contingency table for chi-square test
        all_race_data = []
        for data in interaction_data:
            male_not_got = data["male_total"] - data["male_got"]
            female_not_got = data["female_total"] - data["female_got"]

            if male_not_got < 0:
                print(
                    f"ERROR in {data['race']} matrix: Male not_got negative ({male_not_got})"
                )
                male_not_got = 0
            if female_not_got < 0:
                print(
                    f"ERROR in {data['race']} matrix: Female not_got negative ({female_not_got})"
                )
                female_not_got = 0

            all_race_data.extend(
                [
                    data["male_got"],
                    male_not_got,
                    data["female_got"],
                    female_not_got,
                ]
            )

        # Reshape for chi-square test: rows=races, cols=gender*outcome
        n_races = len(interaction_data)
        if n_races >= 2:
            contingency_matrix = np.array(all_race_data).reshape(n_races, 4)

            # Test if there's an interaction between race and gender
            try:
                from scipy.stats import chi2_contingency

                chi2_stat, p_interaction, dof, expected = chi2_contingency(
                    contingency_matrix
                )

                print(f"\nTEST FOR RACE x GENDER INTERACTION:")
                print(f"Chi-square statistic: {chi2_stat:.4f}")
                print(f"Degrees of freedom: {dof}")
                print(f"P-value: {p_interaction:.6f}")
                print(
                    f"Gender effect varies by race: {'YES' if p_interaction < 0.05 else 'NO'}"
                )

                if p_interaction < 0.05:
                    print(
                        "*** Significant interaction: Gender differences vary significantly between races ***"
                    )
                else:
                    print(
                        "No significant interaction: Gender differences are similar across races"
                    )

            except Exception as e:
                print(f"Could not perform interaction test: {e}")

        # Pairwise comparisons between races for gender differences
        print(f"\nPAIRWISE RACE COMPARISONS FOR GENDER DIFFERENCES:")
        race_pairs = [
            (interaction_data[i], interaction_data[j])
            for i in range(len(interaction_data))
            for j in range(i + 1, len(interaction_data))
        ]

        pairwise_results = []
        for race1_data, race2_data in race_pairs:
            # Test if gender difference in race1 is significantly different from race2
            # Using Breslow-Day test approach for homogeneity of odds ratios

            # Create 2x2x2 table: Race x Gender x Outcome
            table_race1 = [
                [
                    race1_data["male_got"],
                    race1_data["male_total"] - race1_data["male_got"],
                ],
                [
                    race1_data["female_got"],
                    race1_data["female_total"] - race1_data["female_got"],
                ],
            ]
            table_race2 = [
                [
                    race2_data["male_got"],
                    race2_data["male_total"] - race2_data["male_got"],
                ],
                [
                    race2_data["female_got"],
                    race2_data["female_total"] - race2_data["female_got"],
                ],
            ]

            # Ensure non-negative values
            for table in [table_race1, table_race2]:
                for row in table:
                    for i in range(len(row)):
                        if row[i] < 0:
                            row[i] = 0

            try:
                # Test if odds ratios are significantly different between races
                # Combine into single contingency table for comparison
                combined_table = np.array(
                    [
                        [
                            table_race1[0][0],
                            table_race1[0][1],
                            table_race1[1][0],
                            table_race1[1][1],
                        ],  # race1: M_got, M_not, F_got, F_not
                        [
                            table_race2[0][0],
                            table_race2[0][1],
                            table_race2[1][0],
                            table_race2[1][1],
                        ],  # race2: M_got, M_not, F_got, F_not
                    ]
                )

                chi2_pair, p_pair, dof, expected = chi2_contingency(combined_table)

                # Calculate effect sizes
                diff1 = race1_data["gender_diff"]
                diff2 = race2_data["gender_diff"]
                diff_of_diffs = abs(diff1 - diff2)

                pairwise_results.append(
                    {
                        "race1": race1_data["race"],
                        "race2": race2_data["race"],
                        "diff1": diff1,
                        "diff2": diff2,
                        "diff_of_diffs": diff_of_diffs,
                        "p_value": p_pair,
                        "significant": p_pair < 0.05,
                    }
                )

                print(
                    f"  {race1_data['race']:<15} vs {race2_data['race']:<15}: "
                    f"Δ1={diff1:+5.1f}pp Δ2={diff2:+5.1f}pp |Δ1-Δ2|={diff_of_diffs:5.1f}pp p={p_pair:.4f} "
                    f"{'***' if p_pair < 0.05 else ''}"
                )

            except Exception as e:
                print(
                    f"  {race1_data['race']} vs {race2_data['race']}: Could not compare ({e})"
                )

    # 7. Test if age differences vary significantly between races
    print(f"\n" + "-" * 60)
    print("TESTING IF AGE DIFFERENCES VARY BETWEEN RACES:")
    print("-" * 60)

    age_interaction_data = []
    for race in races:
        race_data = comparison[comparison["race_name"] == race]
        young_data = race_data[race_data["age_group"] == "<65"]
        old_data = race_data[race_data["age_group"] == "65+"]

        if not young_data.empty and not old_data.empty:
            young_got = int(young_data["cnt"].sum())
            young_total = int(young_data["base_count"].sum())
            old_got = int(old_data["cnt"].sum())
            old_total = int(old_data["base_count"].sum())

            # Validate data integrity and report issues
            if young_got > young_total:
                print(
                    f"ERROR in {race} age interaction: Young got drugs ({young_got}) > young total ({young_total})"
                )
                young_got = young_total
            if old_got > old_total:
                print(
                    f"ERROR in {race} age interaction: Old got drugs ({old_got}) > old total ({old_total})"
                )
                old_got = old_total

            # Skip if insufficient data
            if young_total == 0 or old_total == 0:
                print(
                    f"WARNING in {race} age interaction: Zero totals - young_total={young_total}, old_total={old_total}"
                )
                continue

            young_rate = (young_got / young_total * 100) if young_total > 0 else 0
            old_rate = (old_got / old_total * 100) if old_total > 0 else 0
            age_diff = old_rate - young_rate

            age_interaction_data.append(
                {
                    "race": race,
                    "young_got": young_got,
                    "young_total": young_total,
                    "young_rate": young_rate,
                    "old_got": old_got,
                    "old_total": old_total,
                    "old_rate": old_rate,
                    "age_diff": age_diff,
                }
            )

    # Display age differences by race
    print("\nAge differences by race:")
    for data in age_interaction_data:
        print(
            f"  {data['race']:<20}: <65={data['young_rate']:5.1f}% 65+={data['old_rate']:5.1f}% Diff={data['age_diff']:+5.1f}pp"
        )

    # Test for homogeneity of age differences across races
    if len(age_interaction_data) >= 2:
        # Calculate odds ratios for each race (65+ vs <65)
        age_odds_ratios = []
        for data in age_interaction_data:
            if data["young_total"] > 0 and data["old_total"] > 0:
                young_odds = data["young_got"] / max(
                    1, data["young_total"] - data["young_got"]
                )
                old_odds = data["old_got"] / max(1, data["old_total"] - data["old_got"])
                if young_odds > 0:
                    or_value = old_odds / young_odds
                    age_odds_ratios.append(or_value)
                    print(f"  {data['race']:<20}: OR (65+ vs <65) = {or_value:.3f}")

        # Create combined contingency table for age interaction test
        all_age_data = []
        for data in age_interaction_data:
            young_not_got = data["young_total"] - data["young_got"]
            old_not_got = data["old_total"] - data["old_got"]

            if young_not_got < 0:
                print(
                    f"ERROR in {data['race']} age matrix: Young not_got negative ({young_not_got})"
                )
                young_not_got = 0
            if old_not_got < 0:
                print(
                    f"ERROR in {data['race']} age matrix: Old not_got negative ({old_not_got})"
                )
                old_not_got = 0

            all_age_data.extend(
                [
                    data["young_got"],
                    young_not_got,
                    data["old_got"],
                    old_not_got,
                ]
            )

        n_races = len(age_interaction_data)
        if n_races >= 2:
            age_contingency_matrix = np.array(all_age_data).reshape(n_races, 4)

            try:
                chi2_stat, p_age_interaction, dof, expected = chi2_contingency(
                    age_contingency_matrix
                )

                print(f"\nTEST FOR RACE x AGE INTERACTION:")
                print(f"Chi-square statistic: {chi2_stat:.4f}")
                print(f"Degrees of freedom: {dof}")
                print(f"P-value: {p_age_interaction:.6f}")
                print(
                    f"Age effect varies by race: {'YES' if p_age_interaction < 0.05 else 'NO'}"
                )

                if p_age_interaction < 0.05:
                    print(
                        "*** Significant interaction: Age differences vary significantly between races ***"
                    )
                else:
                    print(
                        "No significant interaction: Age differences are similar across races"
                    )

            except Exception as e:
                print(f"Could not perform age interaction test: {e}")

        # Pairwise comparisons between races for age differences
        print(f"\nPAIRWISE RACE COMPARISONS FOR AGE DIFFERENCES:")
        age_race_pairs = [
            (age_interaction_data[i], age_interaction_data[j])
            for i in range(len(age_interaction_data))
            for j in range(i + 1, len(age_interaction_data))
        ]

        age_pairwise_results = []
        for race1_data, race2_data in age_race_pairs:
            # Test if age difference in race1 is significantly different from race2

            # Create 2x2x2 table: Race x Age x Outcome
            table_race1 = [
                [
                    race1_data["young_got"],
                    race1_data["young_total"] - race1_data["young_got"],
                ],
                [
                    race1_data["old_got"],
                    race1_data["old_total"] - race1_data["old_got"],
                ],
            ]
            table_race2 = [
                [
                    race2_data["young_got"],
                    race2_data["young_total"] - race2_data["young_got"],
                ],
                [
                    race2_data["old_got"],
                    race2_data["old_total"] - race2_data["old_got"],
                ],
            ]

            # Ensure non-negative values
            for table in [table_race1, table_race2]:
                for row in table:
                    for i in range(len(row)):
                        if row[i] < 0:
                            row[i] = 0

            try:
                # Test if odds ratios are significantly different between races
                combined_table = np.array(
                    [
                        [
                            table_race1[0][0],
                            table_race1[0][1],
                            table_race1[1][0],
                            table_race1[1][1],
                        ],  # race1: Y_got, Y_not, O_got, O_not
                        [
                            table_race2[0][0],
                            table_race2[0][1],
                            table_race2[1][0],
                            table_race2[1][1],
                        ],  # race2: Y_got, Y_not, O_got, O_not
                    ]
                )

                chi2_pair, p_pair, dof, expected = chi2_contingency(combined_table)

                # Calculate effect sizes
                diff1 = race1_data["age_diff"]
                diff2 = race2_data["age_diff"]
                diff_of_diffs = abs(diff1 - diff2)

                age_pairwise_results.append(
                    {
                        "race1": race1_data["race"],
                        "race2": race2_data["race"],
                        "diff1": diff1,
                        "diff2": diff2,
                        "diff_of_diffs": diff_of_diffs,
                        "p_value": p_pair,
                        "significant": p_pair < 0.05,
                    }
                )

                print(
                    f"  {race1_data['race']:<15} vs {race2_data['race']:<15}: "
                    f"Δ1={diff1:+5.1f}pp Δ2={diff2:+5.1f}pp |Δ1-Δ2|={diff_of_diffs:5.1f}pp p={p_pair:.4f} "
                    f"{'***' if p_pair < 0.05 else ''}"
                )

            except Exception as e:
                print(
                    f"  {race1_data['race']} vs {race2_data['race']}: Could not compare ({e})"
                )

    # SUMMARY SECTION
    print(f"\n" + "=" * 80)
    print("COMPREHENSIVE SUMMARY:")
    print("=" * 80)

    print(f"\n1. AGE-STRATIFIED GENDER DIFFERENCES:")
    if age_gender_results:
        age_gender_df = pd.DataFrame(age_gender_results)
        for _, row in age_gender_df.iterrows():
            sig_marker = "***" if row["significant"] else ""
            print(
                f"  {row['age_group']:<8} M-F diff: {row['difference']:+6.2f}pp  (p={row['p_value']:.4f}) {sig_marker}"
            )

    print(f"\n2. GENDER-STRATIFIED AGE DIFFERENCES:")
    if gender_age_results:
        gender_age_df = pd.DataFrame(gender_age_results)
        for _, row in gender_age_df.iterrows():
            sig_marker = "***" if row["significant"] else ""
            print(
                f"  {row['gender']:<8} 65+ vs <65 diff: {row['difference']:+6.2f}pp  (p={row['p_value']:.4f}) {sig_marker}"
            )

    print(f"\n3. RACE-STRATIFIED GENDER DIFFERENCES:")
    if race_results:
        race_df = pd.DataFrame(race_results)
        race_df_sorted = race_df.sort_values("difference", ascending=False)
        for _, row in race_df_sorted.iterrows():
            sig_marker = "***" if row["significant"] else ""
            print(
                f"  {row['race']:<35} {row['difference']:+6.2f}pp  (p={row['p_value']:.4f}) {sig_marker}"
            )

    print(f"\n4. PAIRWISE RACE COMPARISONS - GENDER DIFFERENCES:")
    if "pairwise_results" in locals() and pairwise_results:
        pairwise_df = pd.DataFrame(pairwise_results)
        pairwise_df_sorted = pairwise_df.sort_values("diff_of_diffs", ascending=False)
        for _, row in pairwise_df_sorted.iterrows():
            sig_marker = "***" if row["significant"] else ""
            print(
                f"  {row['race1']:<12} vs {row['race2']:<12}: |Δ1-Δ2|={row['diff_of_diffs']:5.1f}pp (p={row['p_value']:.4f}) {sig_marker}"
            )

    print(f"\n5. PAIRWISE RACE COMPARISONS - AGE DIFFERENCES:")
    if "age_pairwise_results" in locals() and age_pairwise_results:
        age_pairwise_df = pd.DataFrame(age_pairwise_results)
        age_pairwise_df_sorted = age_pairwise_df.sort_values(
            "diff_of_diffs", ascending=False
        )
        for _, row in age_pairwise_df_sorted.iterrows():
            sig_marker = "***" if row["significant"] else ""
            print(
                f"  {row['race1']:<12} vs {row['race2']:<12}: |Δ1-Δ2|={row['diff_of_diffs']:5.1f}pp (p={row['p_value']:.4f}) {sig_marker}"
            )

    # Bonferroni correction for multiple comparisons
    all_tests = []
    if age_gender_results:
        all_tests.extend([r["p_value"] for r in age_gender_results])
    if gender_age_results:
        all_tests.extend([r["p_value"] for r in gender_age_results])
    if race_results:
        all_tests.extend([r["p_value"] for r in race_results])
    if "pairwise_results" in locals() and pairwise_results:
        all_tests.extend([r["p_value"] for r in pairwise_results])
    if "age_pairwise_results" in locals() and age_pairwise_results:
        all_tests.extend([r["p_value"] for r in age_pairwise_results])

    n_tests = len(all_tests)
    bonferroni_alpha = 0.05 / n_tests if n_tests > 0 else 0.05

    print(f"\nMULTIPLE COMPARISONS CORRECTION:")
    print(f"Total tests performed: {n_tests}")
    print(f"Bonferroni-corrected α: {bonferroni_alpha:.6f}")

    significant_after_correction = [p for p in all_tests if p < bonferroni_alpha]
    print(
        f"Tests significant after Bonferroni correction: {len(significant_after_correction)}"
    )

    if len(significant_after_correction) == 0:
        print(
            "No differences remain statistically significant after multiple comparison correction."
        )
