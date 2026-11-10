import pandas as pd
from pathlib import Path
import sys

from sklearn.model_selection import train_test_split

sys.path.append(
    str(Path(__file__).resolve().parent)
)

from config import (
    CSV_FILE,
    PROCESSED_DIR,
    IMAGE_DIR,
    DISEASES
)


# ============================================================
# CONVERT LABELS
# ============================================================

def convert_labels(label_string):

    labels = str(label_string).split("|")

    return {
        disease: int(disease in labels)
        for disease in DISEASES
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n======================================")
    print("       CREATING DATASET")
    print("======================================")

    # --------------------------------------------------------
    # LOAD ORIGINAL CSV
    # --------------------------------------------------------

    print("\nLoading metadata...")

    df = pd.read_csv(
        CSV_FILE,
        dtype={
            "Image Index": str,
            "Finding Labels": str
        }
    )

    print("Original images:", len(df))

    # --------------------------------------------------------
    # REMOVE MISSING IMAGE NAMES
    # --------------------------------------------------------

    df = df.dropna(
        subset=["Image Index"]
    )

    # Make absolutely sure Image Index is string
    df["Image Index"] = (
        df["Image Index"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # CHECK IMAGE FILES
    # --------------------------------------------------------

    print("\nChecking image files...")

    image_exists = df["Image Index"].apply(
        lambda filename:
        (IMAGE_DIR / filename).is_file()
    )

    df = df[image_exists].copy()

    print(
        "Images available:",
        len(df)
    )

    if len(df) == 0:

        raise RuntimeError(
            "No valid images found.\n"
            f"Check IMAGE_DIR:\n{IMAGE_DIR}"
        )

    # --------------------------------------------------------
    # MANAGEABLE SUBSET
    # --------------------------------------------------------

    sample_size = min(
        10000,
        len(df)
    )

    df = df.sample(
        n=sample_size,
        random_state=42
    ).reset_index(drop=True)

    print(
        "Images selected:",
        len(df)
    )

    # --------------------------------------------------------
    # CONVERT DISEASE LABELS
    # --------------------------------------------------------

    print("\nConverting disease labels...")

    label_data = (
        df["Finding Labels"]
        .apply(convert_labels)
    )

    label_df = pd.DataFrame(
        label_data.tolist()
    )

    # --------------------------------------------------------
    # COMBINE IMAGE + LABEL DATA
    # --------------------------------------------------------

    df = pd.concat(
        [
            df[
                [
                    "Image Index",
                    "Finding Labels"
                ]
            ].reset_index(drop=True),

            label_df.reset_index(drop=True)

        ],
        axis=1
    )

    # --------------------------------------------------------
    # TRAIN / TEMP SPLIT
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        shuffle=True
    )

    # --------------------------------------------------------
    # VALIDATION / TEST SPLIT
    # --------------------------------------------------------

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        shuffle=True
    )

    # --------------------------------------------------------
    # RESET INDICES
    # --------------------------------------------------------

    train_df = train_df.reset_index(drop=True)

    validation_df = validation_df.reset_index(drop=True)

    test_df = test_df.reset_index(drop=True)

    # --------------------------------------------------------
    # CREATE PROCESSED DIRECTORY
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SAVE DATASETS
    # --------------------------------------------------------

    train_df.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False
    )

    validation_df.to_csv(
        PROCESSED_DIR / "validation.csv",
        index=False
    )

    test_df.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n======================================")
    print("       DATASET CREATED")
    print("======================================")

    print(
        "\nTrain      :",
        len(train_df)
    )

    print(
        "Validation :",
        len(validation_df)
    )

    print(
        "Test       :",
        len(test_df)
    )

    print(
        "\nFiles saved in:"
    )

    print(
        PROCESSED_DIR
    )

    print("\nSample Image Index:")

    print(
        train_df["Image Index"].head().tolist()
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()