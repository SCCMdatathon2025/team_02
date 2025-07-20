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


def main(ids: list[int]):
    cohort = get_cohort()
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

    query = f"""
/*───────────────── 1. Cohort (already has demo fields) ─────────────────*/
WITH cohort AS (
  SELECT
    person_id,
    visit_occurrence_id,
    race_concept_id,
    race_name,
    gender_concept_id,
    gender_name
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

/*───────────────── 4. Race × Gender counts (no extra joins needed) ────*/
SELECT
    race_name,
    gender_concept_id,
    gender_name,
    COUNT(DISTINCT person_id) AS cnt      -- change to COUNT(*) for visit-level
FROM cohort_with_drug
GROUP BY
    race_name,
    gender_concept_id, gender_name
ORDER BY cnt DESC;
"""

    # Compute number of people in cohort by race and gender
    base_cohort_counts = (
        cohort[["race_name", "gender_name"]].value_counts().reset_index()
    )
    base_cohort_counts.columns = ["race_name", "gender_name", "base_count"]

    # Run the query to get drug-related cohort counts
    results = run(query)

    # Merge the results to compute ratios
    comparison = results.merge(
        base_cohort_counts, on=["race_name", "gender_name"], how="left"
    )

    # Calculate the ratio (drug cohort / base cohort)
    comparison["ratio"] = comparison["cnt"] / comparison["base_count"]
    comparison["percentage"] = comparison["ratio"] * 100

    # Display results
    # print("\nBase cohort counts by race and gender:")
    # print(base_cohort_counts.sort_values("base_count", ascending=False))

    # print("\nDrug-related cohort counts by race and gender:")
    # print(results.sort_values("cnt", ascending=False))

    # print("\nComparison - Drug cohort representation vs Base cohort:")
    # print(
    #     comparison[["race_name", "gender_name", "base_count", "cnt"]].sort_values(
    #         "cnt", ascending=False
    #     )
    # )

    # Calculate percentages and add statistical significance testing
    comparison["percentage"] = (comparison["cnt"] / comparison["base_count"]) * 100

    # Import statistical testing libraries
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from scipy.stats import chi2_contingency, fisher_exact

    print("\nPercentage of people who received drugs by race and gender:")
    print(
        comparison[
            ["race_name", "gender_name", "base_count", "cnt", "percentage"]
        ].sort_values("percentage", ascending=False)
    )

    # 1. Overall gender difference test
    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 60)

    # Aggregate by gender across all races
    gender_summary = (
        comparison.groupby("gender_name")
        .agg({"base_count": "sum", "cnt": "sum"})
        .reset_index()
    )
    gender_summary["percentage"] = (
        gender_summary["cnt"] / gender_summary["base_count"]
    ) * 100

    print("\nOverall gender summary:")
    print(gender_summary)

    # Chi-square test for overall gender difference
    male_data = gender_summary[gender_summary["gender_name"] == "MALE"]
    female_data = gender_summary[gender_summary["gender_name"] == "FEMALE"]

    # Create contingency table: [got_drug, no_drug] x [male, female]
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

    # 2. Gender differences within each race
    print(f"\n" + "-" * 50)
    print("GENDER DIFFERENCES WITHIN EACH RACE:")
    print("-" * 50)

    race_results = []

    for race in comparison["race_name"].unique():
        race_data = comparison[comparison["race_name"] == race]

        if len(race_data) == 2:  # Should have both male and female
            male_row = race_data[race_data["gender_name"] == "MALE"]
            female_row = race_data[race_data["gender_name"] == "FEMALE"]

            if not male_row.empty and not female_row.empty:
                male_got = male_row["cnt"].iloc[0]
                male_total = male_row["base_count"].iloc[0]
                male_rate = male_row["percentage"].iloc[0]

                female_got = female_row["cnt"].iloc[0]
                female_total = female_row["base_count"].iloc[0]
                female_rate = female_row["percentage"].iloc[0]

                # Create contingency table for this race
                contingency_race = [
                    [male_got, male_total - male_got],
                    [female_got, female_total - female_got],
                ]

                # Use Fisher's exact test for smaller samples, chi-square for larger
                if min(male_total, female_total) < 50:
                    # Fisher's exact test
                    oddsratio, p_value = fisher_exact(contingency_race)
                    test_used = "Fisher's exact"
                else:
                    # Chi-square test
                    chi2, p_value, dof, expected = chi2_contingency(contingency_race)
                    test_used = "Chi-square"

                # Calculate difference in percentages
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
                print(
                    f"  Female rate: {female_rate:.2f}% ({female_got}/{female_total})"
                )
                print(f"  Difference (M-F): {diff:+.2f} percentage points")
                print(f"  {test_used} p-value: {p_value:.6f}")
                print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")

    # Summary of race-specific results
    print(f"\n" + "=" * 50)
    print("SUMMARY OF RACE-SPECIFIC GENDER DIFFERENCES:")
    print("=" * 50)

    race_df = pd.DataFrame(race_results)
    race_df_sorted = race_df.sort_values("difference", ascending=False)

    print("\nRanked by gender difference (Male - Female):")
    for _, row in race_df_sorted.iterrows():
        sig_marker = "***" if row["significant"] else ""
        print(
            f"{row['race']:<35} {row['difference']:+6.2f}pp  (p={row['p_value']:.4f}) {sig_marker}"
        )

    # Bonferroni correction for multiple comparisons
    n_tests = len(race_results)
    bonferroni_alpha = 0.05 / n_tests
    significant_after_correction = race_df[race_df["p_value"] < bonferroni_alpha]

    print(f"\nAfter Bonferroni correction (α = {bonferroni_alpha:.4f}):")
    if len(significant_after_correction) > 0:
        print("Races with significant gender differences:")
        for _, row in significant_after_correction.iterrows():
            print(
                f"  {row['race']}: {row['difference']:+.2f}pp (p={row['p_value']:.6f})"
            )
    else:
        print(
            "No race-specific gender differences remain significant after correction."
        )
