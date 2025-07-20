#!/usr/bin/env python3
"""
Simple BigQuery Runner - Just run queries!
"""

query = ""
with open("sample_queries/example.sql", "r") as f:
    query = f.read()

import os

import pandas as pd
from google.cloud import bigquery

# Configuration
PROJECT_ID = "sepsis-nlp"

# Initialize client
client = bigquery.Client(project=PROJECT_ID)


def get_concept_names(concept_ids: list[int]):
    query = f"""
SELECT * FROM `sccm-discovery.rediscover_datathon_2025.concept` WHERE concept_id IN ({",".join([str(i) for i in concept_ids])})
"""
    return run_query(query)


def run_query(sql, key=None):
    """Run a SQL query and return results as DataFrame"""
    if key and os.path.exists(f"{key}.csv"):
        return pd.read_csv(f"{key}.csv")
    query_job = client.query(sql)
    df = query_job.to_dataframe()
    if key:
        df.to_csv(f"{key}.csv", index=False)
    return df


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-file", type=str, default="")

    parser.add_argument("--table-name", type=str, default="")

    args = parser.parse_args()
    file_name = args.save_file
    table_name = args.table_name

    # Run the query
    results = run_query(query, "temp")

    concept_ids = results["ethnicity_concept_id"].unique()
    concept_names = get_concept_names(concept_ids)

    results["ethnicity_name"] = results["ethnicity_concept_id"].map(
        concept_names.set_index("concept_id")["concept_name"]
    )

    mapping = {"No matching concept": "Unknown"}

    for eth in results["ethnicity_name"].unique():
        if eth not in mapping:
            mapping[eth] = eth

    results["ethnicity_name"] = results["ethnicity_name"].map(mapping)

    person_per_site = run_query(
        """
SELECT * FROM `sccm-discovery.rediscover_datathon_2025.person_per_site`
"""
    )

    results["site_id"] = results["person_id"].map(
        person_per_site.set_index("person_id")["src_name"]
    )

    mapping = {
        "SITE-1": "MW",
        "SITE-2": "SE",
        "SITE-3": "SW",
        "SITE-4": "MW",
        "SITE-5": "NE",
        "SITE-6": "NE",
        "SITE-7": "SE",
        "SITE-8": "SE",
    }

    results["site_location"] = results["site_id"].map(mapping)

    print(results["race_name"].value_counts())

    race_map = {
        "American Indian or Alaska Native": "Other",
        "Native Hawaiian or Other Pacific Islander": "Other",
        "Asian": "Asian",
        "Black": "Black",
        "White": "White",
        "Not Reported": "Not Reported",
        "Mixed": "Other",
    }

    results["race_name"] = results["race_name"].map(race_map)

    if file_name.strip():
        results.to_csv(file_name, index=False)

    if table_name.strip():
        results.to_gbq(table_name, project_id=PROJECT_ID, if_exists="replace")
