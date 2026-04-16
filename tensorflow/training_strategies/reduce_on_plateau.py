import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def build_model():
    return keras.Sequential([
        layers.Input(shape=(14,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])


def main():
    x = np.random.rand(600, 14).astype("float32")
    y = (x[:, :7].sum(axis=1) > 3.5).astype("float32")

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-5
    )

    model.fit(
        x,
        y,
        validation_split=0.2,
        epochs=8,
        batch_size=32,
        callbacks=[reduce_lr],
        verbose=1
    )


if __name__ == "__main__":
    main()
