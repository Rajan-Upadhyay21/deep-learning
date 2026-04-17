import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def create_series(length=500):
    time = np.arange(length, dtype="float32")
    return (np.sin(0.05 * time) + 0.2 * np.cos(0.12 * time)).astype("float32")


def make_windows(series, window_size=20, horizon=3):
    x, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        x.append(series[i:i + window_size])
        y.append(series[i + window_size:i + window_size + horizon])
    x = np.array(x, dtype="float32")
    y = np.array(y, dtype="float32")
    return x, y


def build_model(window_size=20, horizon=3):
    return keras.Sequential([
        layers.Input(shape=(window_size,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(horizon)
    ])


def main():
    series = create_series()
    x, y = make_windows(series)

    model = build_model()
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(x, y, epochs=5, batch_size=32, verbose=1)

    preds = model.predict(x[:2], verbose=0)
    print("Predictions:\n", preds)


if __name__ == "__main__":
    main()
