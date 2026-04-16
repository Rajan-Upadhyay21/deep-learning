import tensorflow as tf
from tensorflow import keras

from generator_network import build_generator
from discriminator_network import build_discriminator


def build_gan(latent_dim: int = 100):
    generator = build_generator(latent_dim=latent_dim)
    discriminator = build_discriminator()
    discriminator.trainable = False

    noise = keras.Input(shape=(latent_dim,))
    fake_images = generator(noise)
    predictions = discriminator(fake_images)

    gan_model = keras.Model(noise, predictions, name="gan_model")
    return generator, discriminator, gan_model


if __name__ == "__main__":
    generator, discriminator, gan_model = build_gan()
    gan_model.summary()
