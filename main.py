import os

import pandas as pd
from google.cloud import bigquery

client = bigquery.Client(project="escim-datathon-2")

GLOBAL_DIR = "data_cache"

os.makedirs(GLOBAL_DIR, exist_ok=True)


def run(query: str):
    results = client.query(query)
    return results.to_dataframe()


def get_18_plus_persons():
    if os.path.exists(os.path.join(GLOBAL_DIR, "18_plus_persons.csv")):
        return pd.read_csv(os.path.join(GLOBAL_DIR, "18_plus_persons.csv"))

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

    persons = run(query)
    persons.to_csv(os.path.join(GLOBAL_DIR, "18_plus_persons.csv"), index=False)
    return persons


def get_concept_names(concept_ids: list[int], key: str):
    if os.path.exists(os.path.join(GLOBAL_DIR, f"{key}_concept_names.csv")):
        return pd.read_csv(os.path.join(GLOBAL_DIR, f"{key}_concept_names.csv"))

    query = f"""
SELECT * FROM `sccm-discovery.rediscover_datathon_2025.concept` WHERE concept_id IN ({",".join([str(i) for i in concept_ids])})
"""
    concepts = run(query)
    concepts.to_csv(os.path.join(GLOBAL_DIR, f"{key}_concept_names.csv"), index=False)
    return concepts


def map_race(persons: pd.DataFrame):
    race_map = {
        "White": "White",
        "Race not stated": "Not Reported",
        "Asian Indian": "Asian",
        "African American": "Black",
        "Other": "Not Reported",
        "No matching concept": "Not Reported",
        "American Indian or Alaska Native": "Native American / Pacific Islander",
        "Other Pacific Islander": "Native American / Pacific Islander",
        "More than one race": "Mixed",
        "Black or African American": "Black",
        "Asian": "Asian",
        "American Indian": "Native American / Pacific Islander",
        "Native Hawaiian or Other Pacific Islander": "Native American / Pacific Islander",
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


def main():
    persons = get_18_plus_persons()
    map_race(persons)
    print(persons["race_name"].value_counts())


if __name__ == "__main__":
    main()


race_maps = {
    "a": "White",
    "b": "Black",
    "c": "Asian",
    "d": "Native American / Pacific Islander",
    "e": "",
}
