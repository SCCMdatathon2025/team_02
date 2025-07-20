from analysis.analysis_gender import main as analysis_gender
from analysis.analysis_gender_age import main as analysis_gender_age


def main(name, ids):
    # print("=" * 100)
    # print(f"Running analysis for {name} by gender")
    # print("=" * 100)

    # analysis_gender(ids)

    print("\n\n")
    print("=" * 100)
    print(f"Running analysis for {name} by gender and age")
    print("=" * 100)
    analysis_gender_age(name, ids)
