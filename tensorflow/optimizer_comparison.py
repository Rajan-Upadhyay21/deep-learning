import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_data(samples=1000):
    x = np.random.rand(samples, 10).astype("float32")
    y = (np.sum(x, axis=1) > 5).astype("float32")
    return x, y


def build_model():
    return keras.Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])


def train_with_optimizer(optimizer_name, x, y):
    model = build_model()
    model.compile(
        optimizer=optimizer_name,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    history = model.fit(x, y, epochs=5, batch_size=32, verbose=0)
    final_acc = history.history["accuracy"][-1]
    print(f"{optimizer_name} final accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    x, y = create_data()
    for optimizer in ["sgd", "adam", "rmsprop"]:
        train_with_optimizer(optimizer, x, y)
