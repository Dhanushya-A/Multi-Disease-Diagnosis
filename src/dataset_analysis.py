import pandas as pd
from pathlib import Path
import sys

sys.path.append(
    str(Path(__file__).resolve().parent)
)

from config import (
    CSV_FILE,
    SELECTED_CSV_FILE,
    DISEASES
)


# ============================================================
# 1. LOAD ORIGINAL DATASET
# ============================================================

def load_original_dataset():

    print("Loading original dataset...")

    df = pd.read_csv(
        CSV_FILE
    )

    print(
        f"\nTotal records: {len(df)}"
    )

    print(
        f"Total columns: {len(df.columns)}"
    )

    return df


# ============================================================
# 2. LOAD SELECTED DATASET
# ============================================================

def load_selected_dataset():

    print("\nLoading selected 1,500-image dataset...")

    if not Path(SELECTED_CSV_FILE).exists():

        raise FileNotFoundError(
            f"Selected CSV not found:\n"
            f"{SELECTED_CSV_FILE}"
        )

    df = pd.read_csv(
        SELECTED_CSV_FILE
    )

    print(
        f"\nSelected records: {len(df)}"
    )

    return df


# ============================================================
# 3. ANALYZE DISEASE DISTRIBUTION
# ============================================================

def analyze_labels(df):

    print("\n" + "=" * 60)
    print("DISEASE DISTRIBUTION")
    print("=" * 60)

    disease_counts = {}

    for disease in DISEASES:

        count = df[
            "Finding Labels"
        ].apply(
            lambda x:
            disease in str(x).split("|")
        ).sum()

        disease_counts[disease] = count

    for disease, count in disease_counts.items():

        print(
            f"{disease:25} : {count}"
        )


# ============================================================
# 4. ANALYZE NO FINDING
# ============================================================

def analyze_no_finding(df):

    count = (
        df["Finding Labels"]
        == "No Finding"
    ).sum()

    print(
        "\nNo Finding images:",
        count
    )


# ============================================================
# 5. SHOW MULTI-DISEASE IMAGES
# ============================================================

def analyze_multiple_diseases(df):

    print("\n" + "=" * 60)
    print("MULTI-DISEASE IMAGES")
    print("=" * 60)

    multi_disease = df[
        df["Finding Labels"].apply(
            lambda x:
            len(str(x).split("|")) > 1
        )
    ]

    print(
        "Images containing multiple diseases:",
        len(multi_disease)
    )

    print("\nExamples:")

    print(
        multi_disease[
            ["Image Index", "Finding Labels"]
        ].head(10)
    )


# ============================================================
# 6. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load selected dataset
    # --------------------------------------------------------

    df = load_selected_dataset()

    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------

    print("\nColumns:")

    print(
        df.columns.tolist()
    )

    # --------------------------------------------------------
    # First records
    # --------------------------------------------------------

    print("\nFirst 5 records:")

    print(
        df.head()
    )

    # --------------------------------------------------------
    # Disease analysis
    # --------------------------------------------------------

    analyze_labels(df)

    # --------------------------------------------------------
    # No Finding
    # --------------------------------------------------------

    analyze_no_finding(df)

    # --------------------------------------------------------
    # Multi-disease analysis
    # --------------------------------------------------------

    analyze_multiple_diseases(df)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()