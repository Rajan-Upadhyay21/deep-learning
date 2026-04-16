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
    x = np.random.rand(800, 6).astype("float32")
    y = np.zeros((800,), dtype="float32")
    y[:80] = 1.0

    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    class_weights = {0: 1.0, 1: 6.0}

    model.fit(
        x,
        y,
        validation_split=0.2,
        epochs=5,
        batch_size=32,
        class_weight=class_weights,
        verbose=1
    )


if __name__ == "__main__":
    main()
