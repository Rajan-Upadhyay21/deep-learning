import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_generator(latent_dim: int = 100):
    model = keras.Sequential(
        [
            layers.Input(shape=(latent_dim,)),
            layers.Dense(7 * 7 * 128, activation="relu"),
            layers.Reshape((7, 7, 128)),
            layers.Conv2DTranspose(128, 4, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu"),
            layers.Conv2D(1, 3, padding="same", activation="tanh"),
        ],
        name="generator_network",
    )
    return model


if __name__ == "__main__":
    generator = build_generator()
    noise = tf.random.normal((4, 100))
    images = generator(noise)
    print("Generated image batch shape:", images.shape)
