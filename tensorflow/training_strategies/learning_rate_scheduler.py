import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def schedule(epoch, lr):
    if epoch < 3:
        return lr
    return lr * 0.5


def build_model():
    return keras.Sequential([
        layers.Input(shape=(8,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])


def main():
    x = np.random.rand(500, 8).astype("float32")
    y = (x.mean(axis=1) > 0.5).astype("float32")

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    lr_callback = keras.callbacks.LearningRateScheduler(schedule)

    model.fit(
        x,
        y,
        validation_split=0.2,
        epochs=6,
        batch_size=32,
        callbacks=[lr_callback],
        verbose=1
    )


if __name__ == "__main__":
    main()
