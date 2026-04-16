import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def main():
    strategy = tf.distribute.MirroredStrategy()

    x = np.random.rand(600, 12).astype("float32")
    y = (x.sum(axis=1) > 6).astype("float32")

    with strategy.scope():
        model = keras.Sequential([
            layers.Input(shape=(12,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(x, y, validation_split=0.2, epochs=4, batch_size=32, verbose=1)
    print("Replicas in sync:", strategy.num_replicas_in_sync)


if __name__ == "__main__":
    main()
