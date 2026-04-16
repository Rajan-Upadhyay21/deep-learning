import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_generator(latent_dim=100):
    model = keras.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(784, activation="sigmoid"),
        layers.Reshape((28, 28, 1))
    ])
    return model


def build_discriminator():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    return model


if __name__ == "__main__":
    generator = build_generator()
    discriminator = build_discriminator()

    generator.summary()
    print("\nDiscriminator Summary:\n")
    discriminator.summary()
