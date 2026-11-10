import tensorflow as tf
import numpy as np
import cv2

from pathlib import Path
import sys


# ============================================================
# ADD SRC TO PYTHON PATH
# ============================================================

sys.path.append(
    str(
        Path(__file__).resolve().parent.parent / "src"
    )
)


from config import (
    IMAGE_SIZE,
    DISEASES,
    MODEL_DIR
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model_path = MODEL_DIR / "best_model.keras"

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    model = tf.keras.models.load_model(
        model_path
    )

    return model


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    # Convert PIL image to RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize(
        IMAGE_SIZE
    )

    # Convert to NumPy
    image_array = np.array(
        image
    ).astype(
        np.float32
    )

    # Normalize
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(
    model,
    image
):

    processed_image = preprocess_image(
        image
    )

    probabilities = model.predict(
        processed_image,
        verbose=0
    )[0]

    results = []

    for disease, probability in zip(
        DISEASES,
        probabilities
    ):

        results.append({

            "Disease": disease,

            "Probability": float(
                probability
            ),

            "Percentage": float(
                probability * 100
            )
        })


    # Highest probability first
    results.sort(
        key=lambda x: x["Probability"],
        reverse=True
    )

    return results


# ============================================================
# GET DETECTED CONDITIONS
# ============================================================

def get_detected_conditions(
    results,
    threshold=0.50
):

    detected = [

        result

        for result in results

        if result["Probability"] >= threshold
    ]

    return detected


# ============================================================
# FIND DENSENET BASE MODEL
# ============================================================

def find_base_model(model):

    """
    Find the DenseNet121 sub-model inside
    the main classification model.
    """

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ):

            return layer

    return None


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer(
    base_model
):

    """
    Find the last Conv2D layer inside DenseNet121.
    """

    for layer in reversed(
        base_model.layers
    ):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):

            return layer

    raise ValueError(
        "Could not find a convolutional layer "
        "inside DenseNet121."
    )


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    image,
    disease_index
):

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    processed_image = preprocess_image(
        image
    )


    # --------------------------------------------------------
    # FIND BASE MODEL
    # --------------------------------------------------------

    base_model = find_base_model(
        model
    )

    if base_model is None:

        raise ValueError(
            "DenseNet121 base model could not be found."
        )


    # --------------------------------------------------------
    # FIND LAST CONVOLUTIONAL LAYER
    # --------------------------------------------------------

    last_conv_layer = find_last_conv_layer(
        base_model
    )


    # --------------------------------------------------------
    # CREATE MODEL FOR LAST CONV LAYER
    # --------------------------------------------------------

    # IMPORTANT:
    #
    # We create a model using the DenseNet model's
    # own input and the last convolutional layer output.
    #
    # This avoids manually passing tensors through
    # DenseNet's internal Concatenate layers.

    conv_model = tf.keras.Model(

        inputs=base_model.input,

        outputs=last_conv_layer.output
    )


    # --------------------------------------------------------
    # FORWARD PASS THROUGH COMPLETE MODEL
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        # Watch the input
        tape.watch(
            processed_image
        )

        # Get convolutional feature maps
        feature_maps = conv_model(
            processed_image,
            training=False
        )

        # Get final prediction
        predictions = model(
            processed_image,
            training=False
        )

        # Select requested disease
        class_output = predictions[
            :, disease_index
        ]


    # --------------------------------------------------------
    # GRADIENTS
    # --------------------------------------------------------

    gradients = tape.gradient(
        class_output,
        feature_maps
    )


    if gradients is None:

        raise ValueError(
            "Gradients could not be calculated. "
            "Grad-CAM cannot be generated."
        )


    # --------------------------------------------------------
    # GLOBAL AVERAGE POOLING
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )


    # Remove batch dimension
    feature_maps = feature_maps[0]

    pooled_gradients = pooled_gradients[0]


    # --------------------------------------------------------
    # WEIGHT FEATURE MAPS
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(

        feature_maps * pooled_gradients,

        axis=-1
    )


    # --------------------------------------------------------
    # RELU
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value
        + tf.keras.backend.epsilon()
    )


    # Convert TensorFlow tensor -> NumPy
    heatmap = heatmap.numpy()


    # --------------------------------------------------------
    # RESIZE HEATMAP
    # --------------------------------------------------------

    heatmap = cv2.resize(

        heatmap,

        (
            image.width,
            image.height
        )
    )


    return heatmap


# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    heatmap,
    alpha=0.4
):

    # --------------------------------------------------------
    # PIL -> RGB
    # --------------------------------------------------------

    original = np.array(
        image.convert("RGB")
    )


    # --------------------------------------------------------
    # HEATMAP -> 0-255
    # --------------------------------------------------------

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )


    # --------------------------------------------------------
    # APPLY COLOR MAP
    # --------------------------------------------------------

    heatmap_color = cv2.applyColorMap(

        heatmap_uint8,

        cv2.COLORMAP_JET
    )


    # OpenCV BGR -> RGB
    heatmap_color = cv2.cvtColor(

        heatmap_color,

        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # MAKE SIZES MATCH
    # --------------------------------------------------------

    heatmap_color = cv2.resize(

        heatmap_color,

        (
            original.shape[1],
            original.shape[0]
        )
    )


    # --------------------------------------------------------
    # CREATE OVERLAY
    # --------------------------------------------------------

    overlay = cv2.addWeighted(

        original,

        1 - alpha,

        heatmap_color,

        alpha,

        0
    )


    return overlay


# ============================================================
# GET TOP PREDICTION
# ============================================================

def get_top_prediction(
    results
):

    if not results:

        return None

    return results[0]


# ============================================================
# GET TOP N PREDICTIONS
# ============================================================

def get_top_predictions(
    results,
    n=5
):

    return results[:n]