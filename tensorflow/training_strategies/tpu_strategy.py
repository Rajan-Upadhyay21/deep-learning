import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def get_strategy():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("TPU found.")
        return tf.distribute.TPUStrategy(resolver)
    except Exception:
        print("TPU not available. Falling back to default strategy.")
        return tf.distribute.get_strategy()


def main():
    strategy = get_strategy()

    x = np.random.rand(400, 8).astype("float32")
    y = (x.mean(axis=1) > 0.5).astype("float32")

    with strategy.scope():
        model = keras.Sequential([
            layers.Input(shape=(8,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(x, y, epochs=3, batch_size=32, verbose=1)
    print("Replicas in sync:", strategy.num_replicas_in_sync)


if __name__ == "__main__":
    main()
