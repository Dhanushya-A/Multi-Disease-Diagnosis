import tensorflow as tf
import numpy as np
from pathlib import Path
import sys

sys.path.append(
    str(Path(__file__).resolve().parent)
)

from config import (
    BASE_DIR,
    IMAGE_SIZE,
    DISEASES
)


# ============================================================
# SELECTED IMAGE DIRECTORY
# ============================================================

IMAGE_DIR = (
    BASE_DIR /
    "dataset" /
    "selected_images"
)


# ============================================================
# CHEST X-RAY DATA GENERATOR
# ============================================================

class ChestXrayGenerator(
    tf.keras.utils.Sequence
):

    def __init__(
        self,
        dataframe,
        batch_size=8,
        shuffle=True
    ):

        super().__init__()

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.indices = np.arange(
            len(self.df)
        )

        self.on_epoch_end()


    # ========================================================
    # NUMBER OF BATCHES
    # ========================================================

    def __len__(self):

        return int(
            np.ceil(
                len(self.df) /
                self.batch_size
            )
        )


    # ========================================================
    # GET ONE BATCH
    # ========================================================

    def __getitem__(self, index):

        batch_indices = self.indices[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        batch = self.df.iloc[
            batch_indices
        ]

        images = []

        labels = []


        # ====================================================
        # LOAD EACH IMAGE
        # ====================================================

        for _, row in batch.iterrows():

            # ------------------------------------------------
            # IMAGE PATH
            # ------------------------------------------------

            image_path = (
                IMAGE_DIR /
                row["Image Index"]
            )


            # ------------------------------------------------
            # CHECK IMAGE EXISTS
            # ------------------------------------------------

            if not image_path.exists():

                raise FileNotFoundError(
                    f"Image not found:\n"
                    f"{image_path}"
                )


            # ------------------------------------------------
            # LOAD IMAGE
            # ------------------------------------------------

            image = (
                tf.keras.utils.load_img(
                    image_path,
                    target_size=IMAGE_SIZE,
                    color_mode="rgb"
                )
            )


            # ------------------------------------------------
            # IMAGE TO NUMPY
            # ------------------------------------------------

            image = (
                tf.keras.utils.img_to_array(
                    image
                )
            )


            # ------------------------------------------------
            # NORMALIZE
            # ------------------------------------------------

            image = (
                image.astype(
                    np.float32
                ) / 255.0
            )


            images.append(
                image
            )


            # =================================================
            # MULTI-LABEL VECTOR
            # =================================================

            label_vector = np.zeros(
                len(DISEASES),
                dtype=np.float32
            )


            finding_labels = str(
                row["Finding Labels"]
            )


            # -------------------------------------------------
            # Example:
            #
            # Cardiomegaly|Effusion
            #
            # becomes:
            #
            # [0, 1, 0, 0, 1, ...]
            # -------------------------------------------------

            disease_list = (
                finding_labels
                .split("|")
            )


            for disease in disease_list:

                disease = (
                    disease.strip()
                )


                if disease in DISEASES:

                    disease_index = (
                        DISEASES.index(
                            disease
                        )
                    )


                    label_vector[
                        disease_index
                    ] = 1.0


            labels.append(
                label_vector
            )


        # ====================================================
        # RETURN BATCH
        # ====================================================

        images = np.array(
            images,
            dtype=np.float32
        )

        labels = np.array(
            labels,
            dtype=np.float32
        )


        return (

            tf.convert_to_tensor(
                images,
                dtype=tf.float32
            ),

            tf.convert_to_tensor(
                labels,
                dtype=tf.float32
            )
        )


    # ========================================================
    # SHUFFLE AFTER EACH EPOCH
    # ========================================================

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )