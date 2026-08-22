"""Keras architecture for SeismicML (workstream W5).

Contract (ROADMAP.md §4.5):
Input(input_dim) -> Dense(128, relu) -> BatchNorm -> Dropout(DROPOUT_RATE)
-> Dense(64, relu) -> Dense(64, relu) -> Dense(1, sigmoid, dtype="float32").

Hidden layer widths are Tensor-Core aligned multiples of 64. The global
mixed-precision policy (``mixed_float16``) must be active at call time; the
final classification head always runs in float32 for numeric stability.
"""

import tensorflow as tf

from src.config import DROPOUT_RATE, LEARNING_RATE


def build_4070ti_model(input_dim: int) -> tf.keras.Model:
    """Build and compile the RTX 4070 Ti model (Tensor Core aligned, XLA JIT)."""
    inputs = tf.keras.Input(shape=(input_dim,))

    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model: tf.keras.Model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        jit_compile=True,
    )
    return model
