import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def create_series(length=350):
    time = np.arange(length, dtype="float32")
    return (np.sin(0.09 * time) + 0.05 * np.random.randn(length)).astype("float32")


def make_windows(series, window_size=12):
    x, y = [], []
    for i in range(len(series) - window_size):
        x.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(x, dtype="float32"), np.array(y, dtype="float32")


def build_model(window_size=12):
    return keras.Sequential([
        layers.Input(shape=(window_size,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])


def main():
    series = create_series()
    x, y = make_windows(series)

    model = build_model()
    model.compile(optimizer="adam", loss="mse")
    model.fit(x, y, epochs=4, batch_size=32, verbose=1)

    next_value = model.predict(x[:1], verbose=0)[0][0]
    print("Next-step forecast:", float(next_value))


if __name__ == "__main__":
    main()
