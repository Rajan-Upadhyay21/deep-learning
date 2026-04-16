import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_dcgan_generator(latent_dim: int = 100):
    return keras.Sequential(
        [
            layers.Input(shape=(latent_dim,)),
            layers.Dense(7 * 7 * 256, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Reshape((7, 7, 256)),
            layers.Conv2DTranspose(128, 5, strides=1, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(64, 5, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(1, 5, strides=2, padding="same", activation="tanh"),
        ],
        name="dcgan_generator",
    )


def build_dcgan_discriminator(input_shape=(28, 28, 1)):
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(64, 5, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            layers.Conv2D(128, 5, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1),
        ],
        name="dcgan_discriminator",
    )


if __name__ == "__main__":
    generator = build_dcgan_generator()
    discriminator = build_dcgan_discriminator()

    noise = tf.random.normal((2, 100))
    fake_images = generator(noise)
    logits = discriminator(fake_images)

    print("Fake image shape:", fake_images.shape)
    print("Logits shape:", logits.shape)
