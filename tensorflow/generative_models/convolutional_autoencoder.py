import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_convolutional_autoencoder(input_shape=(28, 28, 1)):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D(2, padding="same", name="encoded_feature_map")(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(encoded)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)
    outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    autoencoder = keras.Model(inputs, outputs, name="convolutional_autoencoder")
    return autoencoder


if __name__ == "__main__":
    model = build_convolutional_autoencoder()
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.summary()
