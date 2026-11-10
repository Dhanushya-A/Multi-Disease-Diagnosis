import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from config import BASE_DIR, IMAGE_SIZE


# ============================================================
# PATHS FOR REDUCED DATASET
# ============================================================

SELECTED_CSV = (
    BASE_DIR /
    "dataset" /
    "selected_images.csv"
)

SELECTED_IMAGE_DIR = (
    BASE_DIR /
    "dataset" /
    "selected_images"
)

PROCESSED_DIR = (
    BASE_DIR /
    "dataset" /
    "processed"
)


# ============================================================
# 1. LOAD AND PREPROCESS ONE IMAGE
# ============================================================

def load_and_preprocess_image(
    image_path,
    image_size=IMAGE_SIZE
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            f"Could not read image: {image_path}"
        )

    # OpenCV uses BGR
    # Convert BGR -> RGB

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize

    image = cv2.resize(
        image,
        image_size
    )

    # Convert to float32

    image = image.astype(
        np.float32
    )

    # Normalize
    # 0-255 -> 0-1

    image = image / 255.0

    return image


# ============================================================
# 2. PREPROCESS MULTIPLE IMAGES
# ============================================================

def preprocess_batch(
    image_paths,
    image_size=IMAGE_SIZE
):

    images = []

    for path in image_paths:

        try:

            image = load_and_preprocess_image(
                path,
                image_size
            )

            images.append(image)

        except Exception as e:

            print(
                f"Error processing {path}: {e}"
            )

    return np.array(
        images,
        dtype=np.float32
    )


# ============================================================
# 3. CREATE TRAIN / VALIDATION / TEST SPLITS
# ============================================================

def create_dataset_splits():

    print("\n======================================")
    print("       DATASET PREPARATION")
    print("======================================")


    # ========================================================
    # CHECK SELECTED CSV
    # ========================================================

    print("\nLoading selected metadata...")

    if not SELECTED_CSV.exists():

        raise FileNotFoundError(
            f"Selected CSV not found:\n"
            f"{SELECTED_CSV}"
        )


    # ========================================================
    # CHECK SELECTED IMAGE DIRECTORY
    # ========================================================

    if not SELECTED_IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Selected image directory not found:\n"
            f"{SELECTED_IMAGE_DIR}"
        )


    # ========================================================
    # LOAD CSV
    # ========================================================

    df = pd.read_csv(
        SELECTED_CSV
    )

    print(
        "Selected CSV records:",
        len(df)
    )


    # ========================================================
    # FIND SELECTED PNG IMAGES
    # ========================================================

    print("\nChecking selected images...")

    available_images = {

        file.name

        for file in
        SELECTED_IMAGE_DIR.glob("*.png")

    }


    print(
        "Selected PNG images:",
        len(available_images)
    )


    if len(available_images) == 0:

        raise ValueError(
            "No PNG images found in selected_images."
        )


    # ========================================================
    # MATCH CSV WITH IMAGES
    # ========================================================

    print("\nMatching metadata with images...")

    df = df[
        df["Image Index"].isin(
            available_images
        )
    ].copy()


    print(
        "Matching records:",
        len(df)
    )


    if len(df) == 0:

        raise ValueError(
            "No matching images found."
        )


    # ========================================================
    # REMOVE EMPTY LABELS
    # ========================================================

    df = df[
        df["Finding Labels"].notna()
    ].copy()


    df = df[
        df["Finding Labels"].str.strip() != ""
    ].copy()


    print(
        "Records with labels:",
        len(df)
    )


    # ========================================================
    # SHUFFLE DATA
    # ========================================================

    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(
        drop=True
    )


    # ========================================================
    # TRAIN = 70%
    # TEMPORARY = 30%
    # ========================================================

    train_df, temp_df = train_test_split(

        df,

        test_size=0.30,

        random_state=42,

        shuffle=True
    )


    # ========================================================
    # VALIDATION = 15%
    # TEST = 15%
    # ========================================================

    validation_df, test_df = train_test_split(

        temp_df,

        test_size=0.50,

        random_state=42,

        shuffle=True
    )


    # ========================================================
    # CREATE PROCESSED DIRECTORY
    # ========================================================

    PROCESSED_DIR.mkdir(

        parents=True,

        exist_ok=True
    )


    # ========================================================
    # SAVE TRAIN CSV
    # ========================================================

    train_path = (
        PROCESSED_DIR /
        "train.csv"
    )


    train_df.to_csv(

        train_path,

        index=False
    )


    # ========================================================
    # SAVE VALIDATION CSV
    # ========================================================

    validation_path = (
        PROCESSED_DIR /
        "validation.csv"
    )


    validation_df.to_csv(

        validation_path,

        index=False
    )


    # ========================================================
    # SAVE TEST CSV
    # ========================================================

    test_path = (
        PROCESSED_DIR /
        "test.csv"
    )


    test_df.to_csv(

        test_path,

        index=False
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n======================================")
    print("       DATASET SPLIT COMPLETED")
    print("======================================")


    print(
        "Total available :",
        len(df)
    )


    print(
        "Training        :",
        len(train_df)
    )


    print(
        "Validation      :",
        len(validation_df)
    )


    print(
        "Testing         :",
        len(test_df)
    )


    print("\nFiles created:")

    print(train_path)

    print(validation_path)

    print(test_path)


# ============================================================
# 4. MAIN
# ============================================================

def main():

    create_dataset_splits()


if __name__ == "__main__":

    main()