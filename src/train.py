import pandas as pd
import tensorflow as tf
from pathlib import Path
import sys

sys.path.append(
    str(Path(__file__).resolve().parent)
)

from config import (
    PROCESSED_DIR,
    MODEL_DIR,
    BATCH_SIZE,
    EPOCHS
)

from data_generator import ChestXrayGenerator
from model import build_model


def main():

    print("\n======================================")
    print("       CHEST X-RAY TRAINING")
    print("======================================")


    # ========================================================
    # LOAD DATASETS
    # ========================================================

    print("\nLoading datasets...")

    train_file = (
        PROCESSED_DIR / "train.csv"
    )

    validation_file = (
        PROCESSED_DIR / "validation.csv"
    )

    if not train_file.exists():

        raise FileNotFoundError(
            f"Training file not found:\n"
            f"{train_file}"
        )

    if not validation_file.exists():

        raise FileNotFoundError(
            f"Validation file not found:\n"
            f"{validation_file}"
        )


    train_df = pd.read_csv(
        train_file
    )

    validation_df = pd.read_csv(
        validation_file
    )


    print(
        "Train      :",
        len(train_df)
    )

    print(
        "Validation :",
        len(validation_df)
    )


    # ========================================================
    # CREATE GENERATORS
    # ========================================================

    print(
        "\nCreating data generators..."
    )


    train_generator = ChestXrayGenerator(
        train_df,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    validation_generator = ChestXrayGenerator(
        validation_df,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    print(
        "Training batches   :",
        len(train_generator)
    )

    print(
        "Validation batches :",
        len(validation_generator)
    )


    # ========================================================
    # TEST FIRST BATCH
    # ========================================================

    print(
        "\nTesting first training batch..."
    )


    x_batch, y_batch = (
        train_generator[0]
    )


    print(
        "Image batch shape :",
        x_batch.shape
    )

    print(
        "Label batch shape :",
        y_batch.shape
    )


    # ========================================================
    # BUILD MODEL
    # ========================================================

    print(
        "\nBuilding DenseNet121..."
    )


    model = build_model()

    model.summary()


    # ========================================================
    # CREATE MODEL DIRECTORY
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # CALLBACKS
    # ========================================================

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(

            filepath=str(
                MODEL_DIR /
                "best_model.keras"
            ),

            monitor="val_auc",

            mode="max",

            save_best_only=True,

            verbose=1
        ),


        tf.keras.callbacks.EarlyStopping(

            monitor="val_auc",

            patience=3,

            mode="max",

            restore_best_weights=True,

            verbose=1
        ),


        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=2,

            min_lr=1e-7,

            verbose=1
        )
    ]


    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "          STARTING TRAINING"
    )

    print(
        "======================================\n"
    )


    history = model.fit(

        train_generator,

        validation_data=validation_generator,

        epochs=EPOCHS,

        callbacks=callbacks,

        verbose=1
    )


    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    final_model = (
        MODEL_DIR /
        "densenet121_final.keras"
    )


    model.save(
        final_model
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "       TRAINING COMPLETED"
    )

    print(
        "======================================"
    )


    print(
        "\nBest model:"
    )

    print(
        MODEL_DIR /
        "best_model.keras"
    )


    print(
        "\nFinal model:"
    )

    print(
        final_model
    )


if __name__ == "__main__":

    main()