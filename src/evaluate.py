import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    roc_auc_score
)

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent)
)

from config import (
    PROCESSED_DIR,
    MODEL_DIR,
    BATCH_SIZE,
    DISEASES
)

from data_generator import ChestXrayGenerator


# ============================================================
# CREATE TRUE LABELS
# ============================================================

def create_true_labels(df):

    labels = []

    for finding in df["Finding Labels"]:

        label_vector = np.zeros(
            len(DISEASES),
            dtype=np.float32
        )

        disease_list = str(
            finding
        ).split("|")

        for disease in disease_list:

            disease = disease.strip()

            if disease in DISEASES:

                index = DISEASES.index(
                    disease
                )

                label_vector[index] = 1.0

        labels.append(
            label_vector
        )

    return np.array(
        labels,
        dtype=np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n======================================")
    print("          MODEL EVALUATION")
    print("======================================")

    # --------------------------------------------------------
    # TEST CSV
    # --------------------------------------------------------

    test_file = (
        PROCESSED_DIR /
        "test.csv"
    )

    if not test_file.exists():

        raise FileNotFoundError(
            f"Test dataset not found:\n"
            f"{test_file}"
        )

    test_df = pd.read_csv(
        test_file
    )

    print(
        "\nTest images:",
        len(test_df)
    )


    # --------------------------------------------------------
    # GENERATOR
    # --------------------------------------------------------

    generator = ChestXrayGenerator(

        test_df,

        batch_size=BATCH_SIZE,

        shuffle=False
    )


    print(
        "Test batches:",
        len(generator)
    )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model_file = (
        MODEL_DIR /
        "best_model.keras"
    )

    if not model_file.exists():

        raise FileNotFoundError(
            f"Trained model not found:\n"
            f"{model_file}"
        )


    print(
        "\nLoading trained model..."
    )

    model = tf.keras.models.load_model(
        model_file
    )


    # --------------------------------------------------------
    # GENERATE PREDICTIONS
    # --------------------------------------------------------

    print(
        "\nGenerating predictions..."
    )

    predictions = model.predict(
        generator,
        verbose=1
    )

    predictions = np.asarray(
        predictions
    )


    # --------------------------------------------------------
    # CREATE TRUE LABELS
    # --------------------------------------------------------

    y_true = create_true_labels(
        test_df
    )


    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    print(
        "\nTrue labels shape:",
        y_true.shape
    )

    print(
        "Predictions shape:",
        predictions.shape
    )


    if len(y_true) != len(predictions):

        raise ValueError(
            "Number of true labels and "
            "predictions do not match."
        )


    # ========================================================
    # BINARY PREDICTIONS
    # ========================================================

    predictions_binary = (
        predictions >= 0.5
    ).astype(int)


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "       CLASSIFICATION REPORT"
    )

    print(
        "======================================\n"
    )


    print(
        classification_report(

            y_true,

            predictions_binary,

            target_names=DISEASES,

            zero_division=0
        )
    )


    # ========================================================
    # ROC-AUC
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "             ROC-AUC"
    )

    print(
        "======================================"
    )


    auc_scores = []


    for i, disease in enumerate(
        DISEASES
    ):

        try:

            score = roc_auc_score(

                y_true[:, i],

                predictions[:, i]
            )

            auc_scores.append(
                score
            )

            print(
                f"{disease:25}: "
                f"{score:.4f}"
            )


        except ValueError:

            print(
                f"{disease:25}: "
                f"Cannot calculate"
            )


    # --------------------------------------------------------
    # MEAN AUC
    # --------------------------------------------------------

    if len(auc_scores) > 0:

        mean_auc = np.mean(
            auc_scores
        )

        print(
            "\nMean ROC-AUC:",
            f"{mean_auc:.4f}"
        )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_dir = (
        Path("results") /
        "metrics"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # Save predictions

    prediction_df = pd.DataFrame(
        predictions,
        columns=DISEASES
    )

    prediction_df.insert(
        0,
        "Image Index",
        test_df["Image Index"].values
    )


    prediction_file = (
        results_dir /
        "test_predictions.csv"
    )


    prediction_df.to_csv(
        prediction_file,
        index=False
    )


    print(
        "\nPredictions saved to:"
    )

    print(
        prediction_file
    )


    print(
        "\n======================================"
    )

    print(
        "        EVALUATION COMPLETED"
    )

    print(
        "======================================"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()