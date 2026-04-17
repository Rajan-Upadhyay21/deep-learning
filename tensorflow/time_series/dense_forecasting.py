import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def create_series(length=400):
    time = np.arange(length, dtype="float32")
    return (0.5 * np.sin(0.07 * time) + 0.3 * np.cos(0.02 * time)).astype("float32")


def make_windows(series, window_size=15):
    x, y = [], []
    for i in range(len(series) - window_size):
        x.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(x, dtype="float32"), np.array(y, dtype="float32")


def build_model(window_size=15):
    return keras.Sequential([
        layers.Input(shape=(window_size,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])


def main():
    series = create_series()
    x, y = make_windows(series)

    model = build_model()
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(x, y, epochs=5, batch_size=32, verbose=1)

    output = model.predict(x[:5], verbose=0)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()
