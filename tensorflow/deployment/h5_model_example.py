import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def build_model():
    return keras.Sequential([
        layers.Input(shape=(8,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])


def main():
    x = np.random.rand(300, 8).astype("float32")
    y = (x.sum(axis=1) > 4).astype("float32")

    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x, y, epochs=2, batch_size=32, verbose=1)

    model.save("classifier_model.h5")
    print("Model saved to classifier_model.h5")

    restored = keras.models.load_model("classifier_model.h5")
    predictions = restored.predict(x[:3], verbose=0)
    print("Sample predictions:\n", predictions)


if __name__ == "__main__":
    main()
