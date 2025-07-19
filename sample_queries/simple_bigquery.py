#!/usr/bin/env python3
"""
Simple BigQuery Runner - Just run queries!
"""

query = ""
with open("sample_queries/example.sql", "r") as f:
    query = f.read()

from google.cloud import bigquery

# Configuration
PROJECT_ID = "escim-datathon-2"

# Initialize client
client = bigquery.Client(project=PROJECT_ID)


def run_query(sql):
    """Run a SQL query and return results as DataFrame"""
    query_job = client.query(sql)
    return query_job.to_dataframe()


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-file", type=str, default="")

    args = parser.parse_args()
    file_name = args.save_file

    # Run the query
    results = run_query(query)
    print(results)

    if file_name.strip():
        results.to_csv(file_name, index=False)
