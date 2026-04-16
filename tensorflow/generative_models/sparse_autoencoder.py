import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_sparse_autoencoder(input_dim: int = 784, latent_dim: int = 32):
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inputs)
    latent = layers.Dense(
        latent_dim,
        activation="relu",
        activity_regularizer=regularizers.l1(1e-4),
        name="sparse_latent",
    )(x)

    x = layers.Dense(256, activation="relu")(latent)
    outputs = layers.Dense(input_dim, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="sparse_autoencoder")
    return model


if __name__ == "__main__":
    model = build_sparse_autoencoder()
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.summary()
