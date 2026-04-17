import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def create_data(samples=700, features=8):
    x = np.random.rand(samples, features).astype("float32")
    y = (2.0 * x[:, 0] + 1.5 * x[:, 1] - 0.8 * x[:, 2] + 0.2).astype("float32")
    return x, y


def build_model(input_dim=8):
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])


def main():
    x, y = create_data()
    model = build_model(input_dim=x.shape[1])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(x, y, epochs=5, batch_size=32, validation_split=0.2, verbose=1)


if __name__ == "__main__":
    main()
