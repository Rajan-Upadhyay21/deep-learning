import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
import numpy as np


def main():
    mixed_precision.set_global_policy("mixed_float16")

    x = np.random.rand(500, 10).astype("float32")
    y = (x.sum(axis=1) > 5).astype("float32")

    model = keras.Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid", dtype="float32")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(x, y, validation_split=0.2, epochs=4, batch_size=32, verbose=1)
    print("Current policy:", mixed_precision.global_policy())


if __name__ == "__main__":
    main()
