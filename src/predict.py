import tensorflow as tf
import numpy as np
from pathlib import Path
import sys

sys.path.append(
    str(Path(__file__).resolve().parent)
)

from config import (
    MODEL_DIR,
    IMAGE_SIZE,
    DISEASES
)


# ============================================================
# PREDICT ONE IMAGE
# ============================================================

def predict_image(image_path, threshold=0.5):

    image_path = Path(image_path)

    model_path = (
        MODEL_DIR /
        "best_model.keras"
    )


    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not model_path.exists():

        raise FileNotFoundError(
            f"\nTrained model not found:\n"
            f"{model_path}\n\n"
            f"Please train the model first."
        )


    # ========================================================
    # CHECK IMAGE
    # ========================================================

    if not image_path.exists():

        raise FileNotFoundError(
            f"\nImage not found:\n"
            f"{image_path}"
        )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "\nLoading trained model..."
    )

    model = tf.keras.models.load_model(
        model_path
    )


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    print(
        "Loading image..."
    )

    image = tf.keras.utils.load_img(

        image_path,

        target_size=IMAGE_SIZE,

        color_mode="rgb"
    )


    # ========================================================
    # IMAGE -> NUMPY
    # ========================================================

    image = tf.keras.utils.img_to_array(
        image
    )


    # ========================================================
    # NORMALIZE
    # ========================================================

    image = (
        image.astype(
            np.float32
        ) / 255.0
    )


    # ========================================================
    # ADD BATCH DIMENSION
    # ========================================================

    image = np.expand_dims(
        image,
        axis=0
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    probabilities = model.predict(

        image,

        verbose=0
    )[0]


    # ========================================================
    # CREATE RESULTS
    # ========================================================

    results = []

    for disease, probability in zip(
        DISEASES,
        probabilities
    ):

        results.append({

            "disease": disease,

            "probability": float(
                probability
            ),

            "percentage": float(
                probability * 100
            ),

            "positive": (
                probability >= threshold
            )
        })


    # ========================================================
    # SORT BY PROBABILITY
    # ========================================================

    results.sort(

        key=lambda x:
        x["probability"],

        reverse=True
    )


    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "       MULTI-DISEASE PREDICTION"
    )

    print(
        "======================================"
    )


    image_path = input(
        "\nEnter image path: "
    ).strip()


    try:

        results = predict_image(
            image_path,
            threshold=0.5
        )


        print(
            "\n======================================"
        )

        print(
            "           PREDICTIONS"
        )

        print(
            "======================================"
        )


        for result in results:

            disease = result[
                "disease"
            ]

            percentage = result[
                "percentage"
            ]

            positive = result[
                "positive"
            ]


            status = (
                "POSITIVE"
                if positive
                else "Negative"
            )


            print(
                f"{disease:25} : "
                f"{percentage:6.2f}% "
                f"({status})"
            )


        # ====================================================
        # DETECTED DISEASES
        # ====================================================

        detected = [

            result["disease"]

            for result in results

            if result["positive"]
        ]


        print(
            "\n======================================"
        )

        print(
            "       DETECTED CONDITIONS"
        )

        print(
            "======================================"
        )


        if detected:

            for disease in detected:

                print(
                    f"✓ {disease}"
                )

        else:

            print(
                "No disease crossed the "
                "50% threshold."
            )


        print(
            "\nPrediction completed."
        )


    except Exception as e:

        print(
            "\nERROR:"
        )

        print(e)