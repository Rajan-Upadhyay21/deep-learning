import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def build_model():
    return keras.Sequential([
        layers.Input(shape=(15,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])


def main():
    x = np.random.rand(500, 15).astype("float32")
    y = (x.mean(axis=1) > 0.5).astype("float32")

    model = build_model()
    optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(x, y, validation_split=0.2, epochs=5, batch_size=32, verbose=1)


if __name__ == "__main__":
    main()
