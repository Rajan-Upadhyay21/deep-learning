import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def create_multivariate_series(length=600):
    time = np.arange(length, dtype="float32")
    feature_1 = np.sin(0.05 * time) + 0.1 * np.random.randn(length).astype("float32")
    feature_2 = np.cos(0.08 * time) + 0.1 * np.random.randn(length).astype("float32")
    target = 0.6 * feature_1 + 0.4 * feature_2
    data = np.stack([feature_1, feature_2], axis=1)
    return data, target.astype("float32")


def make_windows(data, target, window_size=24):
    x, y = [], []
    for i in range(len(data) - window_size):
        x.append(data[i:i + window_size])
        y.append(target[i + window_size])
    return np.array(x, dtype="float32"), np.array(y, dtype="float32")


def build_model(window_size=24, num_features=2):
    return keras.Sequential([
        layers.Input(shape=(window_size, num_features)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])


def main():
    data, target = create_multivariate_series()
    x, y = make_windows(data, target)

    model = build_model()
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(x, y, epochs=5, batch_size=32, verbose=1)

    preds = model.predict(x[:5], verbose=0)
    print("Prediction shape:", preds.shape)


if __name__ == "__main__":
    main()
