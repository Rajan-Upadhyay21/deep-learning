import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def create_data(samples=600, features=10):
    x = np.random.rand(samples, features).astype("float32")
    y = (x[:, :5].sum(axis=1) > 2.5).astype("float32")
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
    model = build_model(input_dim=x.shape[1])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x, y, epochs=4, batch_size=32, validation_split=0.2, verbose=1)


if __name__ == "__main__":
    main()
