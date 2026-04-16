import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_dataset():
    x = np.random.rand(512, 10).astype("float32")
    y = (x.mean(axis=1) > 0.5).astype("float32")
    return tf.data.Dataset.from_tensor_slices((x, y)).shuffle(512).batch(32)


def main():
    strategy = tf.distribute.get_strategy()

    with strategy.scope():
        model = keras.Sequential([
            layers.Input(shape=(10,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    dataset = create_dataset()
    model.fit(dataset, epochs=3, verbose=1)
    print("Devices in strategy:", strategy.num_replicas_in_sync)


if __name__ == "__main__":
    main()
