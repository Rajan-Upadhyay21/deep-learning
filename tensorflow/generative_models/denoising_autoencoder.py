import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def add_noise(inputs, noise_factor=0.2):
    noisy = inputs + noise_factor * tf.random.normal(shape=tf.shape(inputs))
    return tf.clip_by_value(noisy, 0.0, 1.0)


def build_denoising_autoencoder(input_dim: int = 784, latent_dim: int = 64):
    inputs = keras.Input(shape=(input_dim,))
    noisy_inputs = layers.Lambda(lambda x: add_noise(x))(inputs)

    x = layers.Dense(256, activation="relu")(noisy_inputs)
    latent = layers.Dense(latent_dim, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(latent)
    outputs = layers.Dense(input_dim, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="denoising_autoencoder")
    return model


if __name__ == "__main__":
    model = build_denoising_autoencoder()
    model.compile(optimizer="adam", loss="mse")
    model.summary()
