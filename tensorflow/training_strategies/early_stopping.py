import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def build_model(input_dim=10):
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])


def main():
    x = np.random.rand(500, 10).astype("float32")
    y = (x.sum(axis=1) > 5).astype("float32")

    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        x,
        y,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    print("Training stopped after epochs:", len(history.history["loss"]))


if __name__ == "__main__":
    main()
