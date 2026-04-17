import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_series(length=500):
    time = np.arange(length, dtype="float32")
    series = np.sin(0.1 * time) + 0.1 * np.random.randn(length).astype("float32")
    return series


def make_windows(series, window_size=20):
    x, y = [], []
    for i in range(len(series) - window_size):
        x.append(series[i:i + window_size])
        y.append(series[i + window_size])
    x = np.array(x, dtype="float32")[..., np.newaxis]
    y = np.array(y, dtype="float32")
    return x, y


def build_model(window_size=20):
    return keras.Sequential([
        layers.Input(shape=(window_size, 1)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])


def main():
    series = create_series()
    x, y = make_windows(series)

    model = build_model()
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(x, y, epochs=5, batch_size=32, verbose=1)

    prediction = model.predict(x[:3], verbose=0)
    print("Predictions:\n", prediction.flatten())


if __name__ == "__main__":
    main()
