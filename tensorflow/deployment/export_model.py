import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os


def build_model():
    return keras.Sequential([
        layers.Input(shape=(7,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])


def main():
    x = np.random.rand(250, 7).astype("float32")
    y = (x.sum(axis=1) > 3.5).astype("float32")

    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(x, y, epochs=2, batch_size=16, verbose=0)

    export_dir = os.path.join("exported_models", "classifier_v1")
    os.makedirs("exported_models", exist_ok=True)
    model.export(export_dir)

    print("Model exported to:", export_dir)


if __name__ == "__main__":
    main()
