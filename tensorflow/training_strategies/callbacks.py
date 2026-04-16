import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_data(samples=600, features=16):
    x = np.random.rand(samples, features).astype("float32")
    y = (x.mean(axis=1) > 0.5).astype("float32")
    return x, y


def build_model(input_dim=16):
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])


def main():
    x, y = create_data()
    model = build_model()

    callback_list = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            "callback_model.keras",
            save_best_only=True,
            monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1
        )
    ]

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        x,
        y,
        validation_split=0.2,
        epochs=8,
        batch_size=32,
        callbacks=callback_list,
        verbose=1
    )


if __name__ == "__main__":
    main()
