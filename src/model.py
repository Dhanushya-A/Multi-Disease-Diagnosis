import tensorflow as tf

from config import (
    DISEASES,
    IMAGE_SIZE,
    LEARNING_RATE
)


# ============================================================
# BUILD DENSENET121 MODEL
# ============================================================

def build_model():

    # --------------------------------------------------------
    # PRETRAINED DENSENET121
    # --------------------------------------------------------

    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        ),
        name="densenet121"
    )

    # Freeze pretrained layers initially
    base_model.trainable = False


    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    inputs = tf.keras.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        ),
        name="xray_input"
    )


    # --------------------------------------------------------
    # DENSENET FEATURE EXTRACTION
    # --------------------------------------------------------

    x = base_model(
        inputs,
        training=False
    )


    # --------------------------------------------------------
    # GLOBAL AVERAGE POOLING
    # --------------------------------------------------------

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)


    # --------------------------------------------------------
    # DROPOUT
    # --------------------------------------------------------

    x = tf.keras.layers.Dropout(
        0.4,
        name="dropout_1"
    )(x)


    # --------------------------------------------------------
    # FULLY CONNECTED LAYER
    # --------------------------------------------------------

    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        name="dense_256"
    )(x)


    # --------------------------------------------------------
    # DROPOUT
    # --------------------------------------------------------

    x = tf.keras.layers.Dropout(
        0.3,
        name="dropout_2"
    )(x)


    # --------------------------------------------------------
    # MULTI-LABEL OUTPUT
    # --------------------------------------------------------

    outputs = tf.keras.layers.Dense(
        len(DISEASES),
        activation="sigmoid",
        name="disease_predictions"
    )(x)


    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="multi_disease_densenet121"
    )


    # --------------------------------------------------------
    # COMPILE
    # --------------------------------------------------------

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="binary_crossentropy",

        metrics=[

            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),

            tf.keras.metrics.AUC(
                name="auc",
                multi_label=True
            )
        ]
    )


    return model