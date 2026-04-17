import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_data(samples=500, features=10):
    x = np.random.rand(samples, features).astype("float32")
    y = (x.sum(axis=1) > features / 2).astype("float32")
    return x, y


def build_model(input_dim=10):
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])


def main():
    x, y = create_data()
    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x, y, epochs=3, batch_size=32, verbose=1)

    model.save("saved_classifier.keras")
    print("Model saved to saved_classifier.keras")

    loaded_model = keras.models.load_model("saved_classifier.keras")
    loss, acc = loaded_model.evaluate(x, y, verbose=0)
    print(f"Loaded model accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
