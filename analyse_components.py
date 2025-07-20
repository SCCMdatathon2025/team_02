import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.interpolate import griddata
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, shapiro, ttest_ind
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

import os

import torch

os.makedirs("plots", exist_ok=True)

client = None

GLOBAL_DIR = "data_cache"

os.makedirs(GLOBAL_DIR, exist_ok=True)

TITLE = "Genderwise - Number Drugs vs Death"
COLUMN = "gender_name"
TO_COMPARE = ["number_drugs", "died_in_hospital"]
labels = ["No", "Yes"]

# Global flag for heatmap visualization
USE_HEATMAP = False


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

    # Get the variables to compare
    var1, var2 = TO_COMPARE

    if COLUMN is not None:
        # Group by the specified column
        unique_values = cohort[COLUMN].dropna().unique()
        n_groups = len(unique_values)

        # Calculate subplot layout
        n_cols = min(3, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols

        # Create the plot depending on the number rows and columns
        plt.figure(figsize=(n_cols * 5, n_rows * 5))

        # Calculate global y-axis limits for consistent scaling
        global_min = cohort[var1].min()
        global_max = cohort[var1].max()
        y_margin = (global_max - global_min) * 0.05  # 5% margin
        global_ylim = (global_min - y_margin, global_max + y_margin)

        # Calculate global limits for var2 as well (for scatter plots)
        if var2 in binary_cols:
            # For binary columns, use 0 and 1 as limits
            global_min_var2 = 0
            global_max_var2 = 1
            var2_margin = 0.05  # 5% margin
        else:
            global_min_var2 = cohort[var2].min()
            global_max_var2 = cohort[var2].max()
            var2_margin = (global_max_var2 - global_min_var2) * 0.05  # 5% margin
        global_ylim_var2 = (
            global_min_var2 - var2_margin,
            global_max_var2 + var2_margin,
        )

        # Calculate global maximum count for consistent heatmap scaling
        if USE_HEATMAP:
            global_max_count = 0
            for group_value in unique_values:
                group_data = cohort[cohort[COLUMN] == group_value]
                if len(group_data) > 0:
                    count_data = group_data.groupby([var1, var2]).size()
                    if len(count_data) > 0:
                        global_max_count = max(global_max_count, count_data.max())

        for i, group_value in enumerate(unique_values):
            plt.subplot(n_rows, n_cols, i + 1)

            # Filter data for this group
            group_data = cohort[cohort[COLUMN] == group_value]

            # Create box plot of var1 (los) by var2 (died_in_hospital)
            if var2 in binary_cols:
                # Box plot for binary comparison
                died_data = group_data[group_data[var2] == 1][var1].dropna()
                survived_data = group_data[group_data[var2] == 0][var1].dropna()

                box_data = [survived_data, died_data]

                bp = plt.boxplot(box_data, labels=labels, patch_artist=True)
                bp["boxes"][0].set_facecolor("lightblue")
                bp["boxes"][1].set_facecolor("lightcoral")

                plt.title(f"{group_value}\n(n={len(group_data)})")
                plt.ylabel(var1.replace("_", " ").title())
                plt.xticks(rotation=45)

                # Set consistent y-axis limits
                plt.ylim(global_ylim)

                # Add summary statistics
                if len(died_data) > 0 and len(survived_data) > 0:
                    plt.text(
                        0.02,
                        0.98,
                        f"Survived: μ={survived_data.mean():.1f}, n={len(survived_data)}\n"
                        f"Died: μ={died_data.mean():.1f}, n={len(died_data)}",
                        transform=plt.gca().transAxes,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                        fontsize=8,
                    )
            else:
                # Scatter plot for continuous comparison
                if USE_HEATMAP:
                    # Create proper discrete heatmap for count data
                    if len(group_data) > 0:
                        # Create a pivot table for the heatmap with sorting
                        heatmap_data = (
                            group_data.groupby([var1, var2])
                            .size()
                            .unstack(fill_value=0)
                        )

                        # Sort both axes in ascending order
                        heatmap_data = heatmap_data.sort_index(axis=0).sort_index(
                            axis=1
                        )

                        # Create smooth interpolated version
                        if len(heatmap_data) > 1 and len(heatmap_data.columns) > 1:
                            # Get the data for interpolation
                            x_vals = np.array(heatmap_data.columns)
                            y_vals = np.array(heatmap_data.index)
                            z_vals = heatmap_data.values

                            # Create coordinate matrices
                            x_coords, y_coords = np.meshgrid(x_vals, y_vals)

                            # Create a finer grid for interpolation
                            x_fine = np.linspace(x_vals.min(), x_vals.max(), 50)
                            y_fine = np.linspace(y_vals.min(), y_vals.max(), 50)
                            x_fine_mesh, y_fine_mesh = np.meshgrid(x_fine, y_fine)

                            # Flatten for interpolation
                            points = np.column_stack(
                                (x_coords.flatten(), y_coords.flatten())
                            )
                            values = z_vals.flatten()

                            # Interpolate
                            z_interp = griddata(
                                points,
                                values,
                                (x_fine_mesh, y_fine_mesh),
                                method="linear",
                                fill_value=0,
                            )

                            # Create smooth heatmap
                            plt.imshow(
                                z_interp,
                                extent=[
                                    x_vals.min(),
                                    x_vals.max(),
                                    y_vals.min(),
                                    y_vals.max(),
                                ],
                                origin="lower",
                                cmap="YlOrRd",
                                aspect="auto",
                                vmin=0,
                                vmax=global_max_count,
                                interpolation="bilinear",
                            )
                            plt.colorbar(label="Count")

                            # Add contour lines for better readability
                            plt.contour(
                                x_fine_mesh,
                                y_fine_mesh,
                                z_interp,
                                levels=5,
                                colors="white",
                                alpha=0.3,
                                linewidths=0.5,
                            )
                        else:
                            # Fallback to regular heatmap for small data
                            sns.heatmap(
                                heatmap_data,
                                annot=True,
                                fmt="d",
                                cmap="YlOrRd",
                                cbar_kws={"label": "Count"},
                                linewidths=0.5,
                                linecolor="white",
                                square=False,
                                vmin=0,  # Ensure consistent color scaling starts from 0
                                vmax=global_max_count,  # Use global maximum for consistent scaling
                            )

                    plt.xlabel(var2.replace("_", " ").title())
                    plt.ylabel(var1.replace("_", " ").title())
                    plt.title(f"{group_value}\n(n={len(group_data)})")
                else:
                    # Traditional scatter plot
                    plt.scatter(group_data[var1], group_data[var2], alpha=0.6)
                    plt.xlabel(var1.replace("_", " ").title())
                    plt.ylabel(var2.replace("_", " ").title())
                    plt.title(f"{group_value}\n(n={len(group_data)})")

                    # Set consistent axis limits for scatter plots
                    plt.ylim(global_ylim_var2)
                    plt.xlim(global_ylim)

        plt.suptitle(TITLE, fontsize=16, y=0.98)

    else:
        plt.figure(figsize=(15, 10))
        # Analysis on the general dataset
        # Calculate global maximum count for consistent heatmap scaling
        if USE_HEATMAP and var2 not in binary_cols:
            global_max_count = 0
            if len(cohort) > 0:
                count_data = cohort.groupby([var1, var2]).size()
                if len(count_data) > 0:
                    global_max_count = count_data.max()

        # Calculate global limits for var2 (for scatter plots)
        if var2 not in binary_cols:
            global_min_var2 = cohort[var2].min()
            global_max_var2 = cohort[var2].max()
            var2_margin = (global_max_var2 - global_min_var2) * 0.05  # 5% margin
            global_ylim_var2 = (
                global_min_var2 - var2_margin,
                global_max_var2 + var2_margin,
            )

        if var2 in binary_cols:
            # Box plot for binary comparison
            died_data = cohort[cohort[var2] == 1][var1].dropna()
            survived_data = cohort[cohort[var2] == 0][var1].dropna()

            box_data = [survived_data, died_data]

            bp = plt.boxplot(box_data, labels=labels, patch_artist=True)
            bp["boxes"][0].set_facecolor("lightblue")
            bp["boxes"][1].set_facecolor("lightcoral")

            plt.title(TITLE)
            plt.ylabel(var1.replace("_", " ").title())
            plt.xticks(rotation=45)

            # Add summary statistics
            if len(died_data) > 0 and len(survived_data) > 0:
                plt.text(
                    0.02,
                    0.98,
                    f"Survived: μ={survived_data.mean():.1f}, n={len(survived_data)}\n"
                    f"Died: μ={died_data.mean():.1f}, n={len(died_data)}",
                    transform=plt.gca().transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )
        else:
            # Scatter plot for continuous comparison
            if USE_HEATMAP:
                # Create proper discrete heatmap for count data
                if len(cohort) > 0:
                    # Create a pivot table for the heatmap with sorting
                    heatmap_data = (
                        cohort.groupby([var1, var2]).size().unstack(fill_value=0)
                    )

                    # Sort both axes in ascending order
                    heatmap_data = heatmap_data.sort_index(axis=0).sort_index(axis=1)

                    # Create smooth interpolated version
                    if len(heatmap_data) > 1 and len(heatmap_data.columns) > 1:
                        # Get the data for interpolation
                        x_vals = np.array(heatmap_data.columns)
                        y_vals = np.array(heatmap_data.index)
                        z_vals = heatmap_data.values

                        # Create coordinate matrices
                        x_coords, y_coords = np.meshgrid(x_vals, y_vals)

                        # Create a finer grid for interpolation
                        x_fine = np.linspace(x_vals.min(), x_vals.max(), 50)
                        y_fine = np.linspace(y_vals.min(), y_vals.max(), 50)
                        x_fine_mesh, y_fine_mesh = np.meshgrid(x_fine, y_fine)

                        # Flatten for interpolation
                        points = np.column_stack(
                            (x_coords.flatten(), y_coords.flatten())
                        )
                        values = z_vals.flatten()

                        # Interpolate
                        z_interp = griddata(
                            points,
                            values,
                            (x_fine_mesh, y_fine_mesh),
                            method="linear",
                            fill_value=0,
                        )

                        # Create smooth heatmap
                        plt.imshow(
                            z_interp,
                            extent=[
                                x_vals.min(),
                                x_vals.max(),
                                y_vals.min(),
                                y_vals.max(),
                            ],
                            origin="lower",
                            cmap="YlOrRd",
                            aspect="auto",
                            vmin=0,
                            vmax=global_max_count,
                            interpolation="bilinear",
                        )
                        plt.colorbar(label="Count")

                        # Add contour lines for better readability
                        plt.contour(
                            x_fine_mesh,
                            y_fine_mesh,
                            z_interp,
                            levels=5,
                            colors="white",
                            alpha=0.3,
                            linewidths=0.5,
                        )
                    else:
                        # Fallback to regular heatmap for small data
                        sns.heatmap(
                            heatmap_data,
                            annot=True,
                            fmt="d",
                            cmap="YlOrRd",
                            cbar_kws={"label": "Count"},
                            linewidths=0.5,
                            linecolor="white",
                            square=False,
                            vmin=0,  # Ensure consistent color scaling starts from 0
                            vmax=global_max_count,  # Use global maximum for consistent scaling
                        )

                plt.xlabel(var2.replace("_", " ").title())
                plt.ylabel(var1.replace("_", " ").title())
                plt.title(TITLE)
            else:
                # Traditional scatter plot
                plt.scatter(cohort[var1], cohort[var2], alpha=0.6)
                plt.xlabel(var1.replace("_", " ").title())
                plt.ylabel(var2.replace("_", " ").title())
                plt.title(TITLE)

                # Set consistent y-axis limits
                plt.ylim(global_ylim_var2)
                plt.xlim(global_ylim)

    plt.tight_layout()

    # Save the plot
    plot_filename = f"plots/{TITLE}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Plot saved as: {plot_filename}")

    # Print some basic statistics
    print(f"\nDataset overview:")
    print(f"Total records: {len(cohort)}")

    if COLUMN is not None:
        print(f"\nBreakdown by {COLUMN}:")
        breakdown = (
            cohort.groupby(COLUMN)
            .agg(
                {
                    var1: ["count", "mean", "std"],
                    var2: ["sum", "mean"] if var2 in binary_cols else ["mean", "std"],
                }
            )
            .round(2)
        )
        print(breakdown)

    # Overall statistics
    print(f"\nOverall {var1} statistics:")
    print(cohort[var1].describe())

    if var2 in binary_cols:
        print(f"\nOverall {var2} statistics:")
        print(f"Total died: {cohort[var2].sum()} ({cohort[var2].mean()*100:.1f}%)")
        print(
            f"Total survived: {len(cohort) - cohort[var2].sum()} ({(1-cohort[var2].mean())*100:.1f}%)"
        )
    else:
        print(f"\nOverall {var2} statistics:")
        print(cohort[var2].describe())


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

    cohort["number_interventions"] = (
        cohort["nasal_canula_mask"]
        + cohort["hiflo_oximyzer"]
        + cohort["cpap_bipap"]
        + cohort["mechanical_ventilation"]
        + cohort["ecmo"]
    )

    cohort["number_drugs"] = (
        cohort["steroid_flag"]
        + cohort["narcotic_flag"]
        + cohort["sedative_flag"]
        + cohort["vasopressor_flag"]
    )

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
            "number_interventions",
            "number_drugs",
        ]
    ]

    run_clustering(cohort)


if __name__ == "__main__":
    main()
