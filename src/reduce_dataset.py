import pandas as pd
import shutil
from pathlib import Path

from config import CSV_FILE, IMAGE_DIR, BASE_DIR


# ============================================================
# SETTINGS
# ============================================================

TARGET_IMAGES = 1500

SELECTED_DIR = BASE_DIR / "dataset" / "selected_images"


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n======================================")
    print("       REDUCING DATASET")
    print("======================================")

    # Load CSV
    df = pd.read_csv(CSV_FILE)

    print("Total CSV records:", len(df))

    # Find available images
    available_images = {
        file.name
        for file in Path(IMAGE_DIR).glob("*.png")
    }

    print("Local images:", len(available_images))

    # Keep only records for images that exist
    df = df[
        df["Image Index"].isin(available_images)
    ].copy()

    print("Matching records:", len(df))

    # Remove empty labels
    df = df[
        df["Finding Labels"].notna()
    ].copy()

    # Shuffle
    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # Select 1,500
    selected_df = df.head(TARGET_IMAGES)

    print(
        "Images selected:",
        len(selected_df)
    )

    # Create directory
    SELECTED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Copy images
    print("\nCopying images...")

    copied = 0

    for _, row in selected_df.iterrows():

        image_name = row["Image Index"]

        source = Path(IMAGE_DIR) / image_name

        destination = SELECTED_DIR / image_name

        if source.exists():

            shutil.copy2(
                source,
                destination
            )

            copied += 1

    print("Images copied:", copied)

    # Save selected metadata
    selected_csv = (
        BASE_DIR /
        "dataset" /
        "selected_images.csv"
    )

    selected_df.to_csv(
        selected_csv,
        index=False
    )

    print("\n======================================")
    print("       DATASET REDUCTION COMPLETE")
    print("======================================")

    print("Selected images :", copied)

    print(
        "Selected CSV    :",
        selected_csv
    )

    print(
        "Image directory :",
        SELECTED_DIR
    )


if __name__ == "__main__":
    main()