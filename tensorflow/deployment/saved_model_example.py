import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def build_model():
    return keras.Sequential([
        layers.Input(shape=(12,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])


def main():
    x = np.random.rand(400, 12).astype("float32")
    y = (x.mean(axis=1) > 0.5).astype("float32")

    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x, y, epochs=2, batch_size=32, verbose=1)

    export_path = "saved_model_export"
    model.export(export_path)
    print(f"SavedModel exported to {export_path}")

    loaded = tf.saved_model.load(export_path)
    print("Available signatures:", list(loaded.signatures.keys()))


if __name__ == "__main__":
    main()
