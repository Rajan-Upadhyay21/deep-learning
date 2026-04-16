import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_discriminator(input_shape=(28, 28, 1)):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(64, 4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            layers.Conv2D(128, 4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator_network",
    )
    return model


if __name__ == "__main__":
    discriminator = build_discriminator()
    dummy_images = tf.random.normal((4, 28, 28, 1))
    predictions = discriminator(dummy_images)
    print("Prediction shape:", predictions.shape)
