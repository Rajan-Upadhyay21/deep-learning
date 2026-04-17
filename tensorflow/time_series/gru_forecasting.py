import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def create_series(length=500):
    time = np.arange(length, dtype="float32")
    return (np.cos(0.05 * time) + 0.15 * np.random.randn(length)).astype("float32")


def make_windows(series, window_size=18):
    x, y = [], []
    for i in range(len(series) - window_size):
        x.append(series[i:i + window_size])
        y.append(series[i + window_size])
    x = np.array(x, dtype="float32")[..., np.newaxis]
    y = np.array(y, dtype="float32")
    return x, y


def build_model(window_size=18):
    return keras.Sequential([
        layers.Input(shape=(window_size, 1)),
        layers.GRU(64),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])


def main():
    series = create_series()
    x, y = make_windows(series)

    model = build_model()
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(x, y, epochs=5, batch_size=32, verbose=1)

    preds = model.predict(x[:4], verbose=0)
    print("Prediction shape:", preds.shape)


if __name__ == "__main__":
    main()
