import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(input_dim: int = 784, latent_dim: int = 64):
    encoder_inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(encoder_inputs)
    x = layers.Dense(128, activation="relu")(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent_vector")(x)

    x = layers.Dense(128, activation="relu")(latent)
    x = layers.Dense(256, activation="relu")(x)
    decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x)

    autoencoder = keras.Model(encoder_inputs, decoder_outputs, name="autoencoder")
    encoder = keras.Model(encoder_inputs, latent, name="encoder")
    return autoencoder, encoder


if __name__ == "__main__":
    model, encoder = build_autoencoder()
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    print("\nEncoder summary:\n")
    encoder.summary()
