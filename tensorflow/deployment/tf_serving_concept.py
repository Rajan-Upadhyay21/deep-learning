import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def build_model():
    return keras.Sequential([
        layers.Input(shape=(6,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])


def main():
    x = np.random.rand(250, 6).astype("float32")
    y = (x.sum(axis=1) > 3).astype("float32")

    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(x, y, epochs=2, batch_size=32, verbose=0)

    export_path = "serving_ready_model"
    model.export(export_path)

    print("Model exported for serving.")
    print("Export path:", export_path)
    print("This folder can be used with TensorFlow Serving in a deployment workflow.")


if __name__ == "__main__":
    main()
