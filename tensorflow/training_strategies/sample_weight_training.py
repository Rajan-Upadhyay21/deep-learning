import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def build_model():
    return keras.Sequential([
        layers.Input(shape=(6,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])


def main():
    x = np.random.rand(500, 6).astype("float32")
    y = (x.sum(axis=1) > 3).astype("float32")

    sample_weights = np.ones((500,), dtype="float32")
    sample_weights[:100] = 3.0

    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(
        x,
        y,
        sample_weight=sample_weights,
        validation_split=0.2,
        epochs=5,
        batch_size=32,
        verbose=1
    )


if __name__ == "__main__":
    main()
