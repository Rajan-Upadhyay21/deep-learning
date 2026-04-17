import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def create_series(length=500):
    time = np.arange(length, dtype="float32")
    return (np.sin(0.04 * time) + np.cos(0.1 * time)).astype("float32")


def make_windows(series, window_size=24):
    x, y = [], []
    for i in range(len(series) - window_size):
        x.append(series[i:i + window_size])
        y.append(series[i + window_size])
    x = np.array(x, dtype="float32")[..., np.newaxis]
    y = np.array(y, dtype="float32")
    return x, y


def build_model(window_size=24):
    return keras.Sequential([
        layers.Input(shape=(window_size, 1)),
        layers.Conv1D(32, kernel_size=3, activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])


def main():
    series = create_series()
    x, y = make_windows(series)

    model = build_model()
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(x, y, epochs=5, batch_size=32, verbose=1)
    print("Sample forecast:", model.predict(x[:1], verbose=0).flatten())


if __name__ == "__main__":
    main()
