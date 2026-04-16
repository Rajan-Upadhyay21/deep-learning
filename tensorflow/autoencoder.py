import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(input_dim=784, encoding_dim=64):
    inputs = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(256, activation="relu")(inputs)
    encoded = layers.Dense(encoding_dim, activation="relu")(encoded)

    decoded = layers.Dense(256, activation="relu")(encoded)
    decoded = layers.Dense(input_dim, activation="sigmoid")(decoded)

    autoencoder = keras.Model(inputs, decoded)
    encoder = keras.Model(inputs, encoded)
    return autoencoder, encoder


if __name__ == "__main__":
    autoencoder, encoder = build_autoencoder()
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.summary()
    print("\nEncoder summary:\n")
    encoder.summary()
