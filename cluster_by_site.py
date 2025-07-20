import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, shapiro, ttest_ind
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

import os

import torch

os.makedirs("plots", exist_ok=True)

client = None

GLOBAL_DIR = "data_cache"

os.makedirs(GLOBAL_DIR, exist_ok=True)

TITLE = "Gender"
COLUMN = "gender_name"


def run(query: str, key: str | None = None):
    # If key is provided, check cache first
    if key and os.path.exists(os.path.join(GLOBAL_DIR, f"{key}.csv")):
        return pd.read_csv(os.path.join(GLOBAL_DIR, f"{key}.csv"))

    from google.cloud import bigquery

    global client

    if client is None:
        client = bigquery.Client(project="sepsis-nlp")

    print(f"Executing query for {key}")

    # Execute query
    results = client.query(query)
    df = results.to_dataframe()

    # Save to cache if key is provided
    if key:
        df.to_csv(os.path.join(GLOBAL_DIR, f"{key}.csv"), index=False)

    return df


def get_cohort() -> pd.DataFrame:
    query = """
SELECT * FROM `sepsis-nlp.team_2.cohort_flags_final`
"""
    return run(query, "cohort_cluster")


def preprocess_data(cohort: pd.DataFrame):
    """
    Preprocess data for clustering with mixed data types
    """
    # Separate categorical and numerical columns
    categorical_cols = ["race_name", "gender_name", "site_location", "ethnicity_name"]
    binary_cols = [
        "died_in_hospital",
        "died_in_30_days",
        "steroid_flag",
        "narcotic_flag",
        "sedative_flag",
        "vasopressor_flag",
    ]
    numerical_cols = ["age", "los"]

    # Handle missing values
    cohort_clean = cohort.copy()

    # For categorical columns, fill with mode
    for col in categorical_cols:
        cohort_clean[col] = cohort_clean[col].fillna(cohort_clean[col].mode()[0])

    # For numerical columns, fill with median
    for col in numerical_cols:
        cohort_clean[col] = cohort_clean[col].fillna(cohort_clean[col].median())

    # For binary columns, fill with 0
    for col in binary_cols:
        cohort_clean[col] = cohort_clean[col].fillna(0)

    return cohort_clean, categorical_cols, binary_cols, numerical_cols


def encode_categorical_data(cohort: pd.DataFrame, categorical_cols: list):
    """
    Create different encodings for different clustering algorithms
    """
    cohort_encoded = cohort.copy()

    # Label encoding for categorical variables (for K-Prototypes)
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        cohort_encoded[col + "_encoded"] = le.fit_transform(cohort_encoded[col])
        label_encoders[col] = le

    # One-hot encoding (for traditional algorithms)
    cohort_onehot = pd.get_dummies(
        cohort, columns=categorical_cols, prefix=categorical_cols
    )

    return cohort_encoded, cohort_onehot, label_encoders


def perform_statistical_tests(cohort: pd.DataFrame, column: str):
    """
    Perform statistical tests to compare groups defined by the column variable
    """
    print("\n" + "=" * 80)
    print(f"STATISTICAL SIGNIFICANCE TESTS - COMPARING {column.upper()} GROUPS")
    print("=" * 80)

    # Get unique categories
    categories = cohort[column].unique()
    categories = [cat for cat in categories if pd.notna(cat)]  # Remove NaN values

    if len(categories) < 2:
        print("❌ Cannot perform statistical tests: Need at least 2 categories")
        return

    print(f"📊 Comparing {len(categories)} groups: {', '.join(map(str, categories))}")

    # Define variables to test
    numerical_cols = ["age", "los"]
    binary_cols = [
        "died_in_hospital",
        "died_in_30_days",
        "steroid_flag",
        "narcotic_flag",
        "sedative_flag",
        "vasopressor_flag",
        "nasal_canula_mask",
        "hiflo_oximyzer",
        "cpap_bipap",
        "mechanical_ventilation",
        "ecmo",
        "dialysis",
    ]

    significant_results = []

    # Test numerical variables
    print("\n🔢 NUMERICAL VARIABLES - Statistical Tests:")
    print("-" * 60)

    for col in numerical_cols:
        if col not in cohort.columns:
            continue

        # Get data for each group
        groups = []
        group_names = []
        for cat in categories:
            group_data = cohort[cohort[column] == cat][col].dropna()
            if len(group_data) > 0:
                groups.append(group_data)
                group_names.append(str(cat))

        if len(groups) < 2:
            continue

        # For 2 groups, use appropriate test
        if len(groups) == 2:
            group1, group2 = groups[0], groups[1]
            name1, name2 = group_names[0], group_names[1]

            # Check normality (if sample size allows)
            if len(group1) >= 8 and len(group2) >= 8:
                _, p_norm1 = shapiro(group1.sample(min(5000, len(group1))))
                _, p_norm2 = shapiro(group2.sample(min(5000, len(group2))))
                normal_dist = p_norm1 > 0.05 and p_norm2 > 0.05
            else:
                normal_dist = False

            # Choose appropriate test
            if normal_dist and len(group1) >= 30 and len(group2) >= 30:
                # Use t-test for normal distributions with adequate sample size
                stat, p_value = ttest_ind(group1, group2)
                test_name = "Independent t-test"
            else:
                # Use Mann-Whitney U test for non-normal or small samples
                stat, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
                test_name = "Mann-Whitney U test"

            # Calculate effect size (Cohen's d for t-test, or mean difference for Mann-Whitney)
            mean1, mean2 = group1.mean(), group2.mean()
            std_pooled = np.sqrt(
                (
                    (len(group1) - 1) * group1.std() ** 2
                    + (len(group2) - 1) * group2.std() ** 2
                )
                / (len(group1) + len(group2) - 2)
            )
            cohens_d = abs(mean1 - mean2) / std_pooled if std_pooled > 0 else 0

            # Determine significance
            significance = (
                "***"
                if p_value < 0.001
                else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            )

            if p_value < 0.05:
                significant_results.append(
                    {
                        "variable": col,
                        "test": test_name,
                        "p_value": p_value,
                        "effect_size": cohens_d,
                        "groups": f"{name1} vs {name2}",
                    }
                )

            print(f"   {col.replace('_', ' ').title():20}: {test_name}")
            print(f"      {name1} mean: {mean1:.2f}, {name2} mean: {mean2:.2f}")
            print(
                f"      p-value: {p_value:.4f} {significance}, Cohen's d: {cohens_d:.3f}"
            )

        else:
            # For more than 2 groups, use Kruskal-Wallis test
            stat, p_value = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis test"

            significance = (
                "***"
                if p_value < 0.001
                else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            )

            if p_value < 0.05:
                significant_results.append(
                    {
                        "variable": col,
                        "test": test_name,
                        "p_value": p_value,
                        "effect_size": "N/A",
                        "groups": f"All {len(groups)} groups",
                    }
                )

            print(f"   {col.replace('_', ' ').title():20}: {test_name}")
            print(f"      p-value: {p_value:.4f} {significance}")

    # Test binary/categorical variables
    print("\n🏥 BINARY VARIABLES - Chi-Square Tests:")
    print("-" * 60)

    for col in binary_cols:
        if col not in cohort.columns:
            continue

        # Create contingency table
        contingency_table = pd.crosstab(cohort[column], cohort[col])

        # Skip if any dimension is too small
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            continue

        # Check if expected frequencies are adequate for chi-square
        if contingency_table.min().min() >= 5 and (contingency_table >= 5).all().all():
            # Use chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            test_name = "Chi-square test"

            # Calculate Cramer's V (effect size)
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            effect_size = cramers_v

        else:
            # Use Fisher's exact test (for 2x2 tables) or chi-square with Yates correction
            if contingency_table.shape == (2, 2):
                odds_ratio, p_value = fisher_exact(contingency_table)
                test_name = "Fisher's exact test"
                effect_size = odds_ratio
            else:
                chi2, p_value, dof, expected = chi2_contingency(
                    contingency_table, lambda_=None
                )
                test_name = "Chi-square test (Yates corrected)"
                n = contingency_table.sum().sum()
                cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                effect_size = cramers_v

        significance = (
            "***"
            if p_value < 0.001
            else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        )

        if p_value < 0.05:
            significant_results.append(
                {
                    "variable": col,
                    "test": test_name,
                    "p_value": p_value,
                    "effect_size": effect_size,
                    "groups": f"Groups by {column}",
                }
            )

        # Show proportions for each group
        proportions = contingency_table.div(contingency_table.sum(axis=1), axis=0)

        print(
            f"   {col.replace('_flag', '').replace('_', ' ').title():20}: {test_name}"
        )
        for idx, cat in enumerate(categories):
            if cat in proportions.index and 1 in proportions.columns:
                prop = proportions.loc[cat, 1] if 1 in proportions.columns else 0
                print(f"      {str(cat):10}: {prop:.1%}")
        print(f"      p-value: {p_value:.4f} {significance}")

    # Summary of significant results
    print("\n" + "=" * 80)
    print("📋 SUMMARY OF STATISTICALLY SIGNIFICANT DIFFERENCES")
    print("=" * 80)

    if significant_results:
        print(
            f"Found {len(significant_results)} statistically significant differences (p < 0.05):\n"
        )

        for i, result in enumerate(significant_results, 1):
            sig_level = (
                "***"
                if result["p_value"] < 0.001
                else "**" if result["p_value"] < 0.01 else "*"
            )
            print(f"{i:2d}. {result['variable'].replace('_', ' ').title()}")
            print(f"     Test: {result['test']}")
            print(f"     p-value: {result['p_value']:.4f} {sig_level}")
            if result["effect_size"] != "N/A":
                print(f"     Effect size: {result['effect_size']:.3f}")
            print(f"     Groups: {result['groups']}")
            print()

        # Provide interpretation guidance
        print("🔍 INTERPRETATION GUIDE:")
        print("   * p < 0.05: Significant difference")
        print("   ** p < 0.01: Highly significant difference")
        print("   *** p < 0.001: Very highly significant difference")
        print("   Effect sizes: Small (0.2), Medium (0.5), Large (0.8+) for Cohen's d")
        print("   Effect sizes: Small (0.1), Medium (0.3), Large (0.5+) for Cramer's V")

    else:
        print(
            "❌ No statistically significant differences found between groups (p ≥ 0.05)"
        )

    print("\n" + "=" * 80)


def run_clustering(cohort: pd.DataFrame):
    # Define numerical and binary columns for statistics
    numerical_cols = ["age", "los"]
    binary_cols = [
        "died_in_hospital",
        "died_in_30_days",
        "steroid_flag",
        "narcotic_flag",
        "sedative_flag",
        "vasopressor_flag",
        "nasal_canula_mask",
        "hiflo_oximyzer",
        "cpap_bipap",
        "mechanical_ventilation",
        "ecmo",
        "dialysis",
    ]

    print("=" * 80)
    print(f"COHORT STATISTICS BY {TITLE.upper()}")
    print("=" * 80)

    # for location in cohort["site_location"].unique():
    for location in cohort[COLUMN].unique():
        # cohort_location = cohort[cohort["site_location"] == location]
        cohort_location = cohort[cohort[COLUMN] == location]

        print(f"\n📍 {str(location).upper()}")
        print(f"   Sample size: {len(cohort_location)} patients")
        print("-" * 60)

        # Statistics for numerical columns
        print("\n🔢 NUMERICAL VARIABLES:")
        for col in numerical_cols:
            if col in cohort_location.columns:
                mean_val = cohort_location[col].mean()
                median_val = cohort_location[col].median()
                std_val = cohort_location[col].std()

                if col == "age":
                    print(
                        f"   Age (years):        Mean={mean_val:.1f}, Median={median_val:.1f}, Std={std_val:.1f}"
                    )
                elif col == "los":
                    print(
                        f"   Length of Stay (hrs): Mean={mean_val:.1f}, Median={median_val:.1f}, Std={std_val:.1f}"
                    )
                else:
                    print(
                        f"   {col.replace('_', ' ').title()}: Mean={mean_val:.2f}, Median={median_val:.2f}, Std={std_val:.2f}"
                    )

        # Statistics for binary/flag columns (showing proportions)
        print("\n🏥 CLINICAL FLAGS (Proportions):")
        for col in binary_cols:
            if col in cohort_location.columns:
                proportion = cohort_location[col].mean()
                count = cohort_location[col].sum()
                total = len(cohort_location)

                # Format column name nicely
                formatted_name = col.replace("_flag", "").replace("_", " ").title()
                if "Flag" not in formatted_name and col.endswith("_flag"):
                    formatted_name += " Usage"

                print(f"   {formatted_name:25}: {proportion:.2%} ({count}/{total})")

        print("\n" + "=" * 60)

    # Add statistical analysis
    perform_statistical_tests(cohort, COLUMN)


def main():
    cohort = get_cohort()

    # Convert to datetime and ensure timezone-naive
    cohort["visit_start_datetime"] = pd.to_datetime(cohort["visit_start_datetime"])
    if cohort["visit_start_datetime"].dt.tz is not None:
        cohort["visit_start_datetime"] = cohort["visit_start_datetime"].dt.tz_convert(
            None
        )

    cohort["visit_end_datetime"] = pd.to_datetime(cohort["visit_end_datetime"])
    if cohort["visit_end_datetime"].dt.tz is not None:
        cohort["visit_end_datetime"] = cohort["visit_end_datetime"].dt.tz_convert(None)

    cohort["birth_datetime"] = pd.to_datetime(cohort["birth_datetime"])
    if cohort["birth_datetime"].dt.tz is not None:
        cohort["birth_datetime"] = cohort["birth_datetime"].dt.tz_convert(None)

    # In years
    cohort["age"] = (
        cohort["visit_start_datetime"] - cohort["birth_datetime"]
    ).dt.total_seconds() / (365 * 24 * 60 * 60)

    # In hours
    cohort["los"] = (
        cohort["visit_end_datetime"] - cohort["visit_start_datetime"]
    ).dt.total_seconds() / 3600

    cohort["medicare"] = cohort["age"] >= 65

    cohort = cohort[
        [
            "race_name",
            "ethnicity_name",
            "gender_name",
            "died_in_hospital",
            "died_in_30_days",
            "steroid_flag",
            "narcotic_flag",
            "sedative_flag",
            "vasopressor_flag",
            "los",
            "age",
            "site_location",
            "nasal_canula_mask",
            "hiflo_oximyzer",
            "cpap_bipap",
            "mechanical_ventilation",
            "ecmo",
            "dialysis",
            "medicare",
        ]
    ]

    run_clustering(cohort)


if __name__ == "__main__":
    main()
