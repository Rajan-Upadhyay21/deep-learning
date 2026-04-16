import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_data():
    x = np.random.rand(400, 12).astype("float32")
    y = (x[:, :6].sum(axis=1) > 3).astype("float32")
    return x, y


def build_model():
    return keras.Sequential([
        layers.Input(shape=(12,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])


def main():
    x, y = create_data()
    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath="best_checkpoint.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )

    model.fit(
        x,
        y,
        validation_split=0.2,
        epochs=6,
        batch_size=32,
        callbacks=[checkpoint],
        verbose=1
    )

    restored = keras.models.load_model("best_checkpoint.keras")
    loss, acc = restored.evaluate(x, y, verbose=0)
    print(f"Restored model accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
