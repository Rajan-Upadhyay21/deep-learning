import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_conditional_generator(latent_dim=100, num_classes=10):
    noise_input = keras.Input(shape=(latent_dim,))
    label_input = keras.Input(shape=(1,), dtype="int32")

    label_embedding = layers.Embedding(num_classes, latent_dim)(label_input)
    label_embedding = layers.Flatten()(label_embedding)

    merged = layers.Concatenate()([noise_input, label_embedding])
    x = layers.Dense(7 * 7 * 128, activation="relu")(merged)
    x = layers.Reshape((7, 7, 128))(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2DTranspose(1, 4, strides=2, padding="same", activation="tanh")(x)

    return keras.Model([noise_input, label_input], outputs, name="conditional_generator")


def build_conditional_discriminator(num_classes=10, input_shape=(28, 28, 1)):
    image_input = keras.Input(shape=input_shape)
    label_input = keras.Input(shape=(1,), dtype="int32")

    label_embedding = layers.Embedding(num_classes, 28 * 28)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    label_embedding = layers.Reshape((28, 28, 1))(label_embedding)

    merged = layers.Concatenate()([image_input, label_embedding])
    x = layers.Conv2D(64, 4, strides=2, padding="same")(merged)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model([image_input, label_input], outputs, name="conditional_discriminator")


if __name__ == "__main__":
    generator = build_conditional_generator()
    discriminator = build_conditional_discriminator()

    noise = tf.random.normal((4, 100))
    labels = tf.constant([[1], [3], [5], [7]], dtype=tf.int32)
    images = generator([noise, labels])
    scores = discriminator([images, labels])

    print("Generated shape:", images.shape)
    print("Scores shape:", scores.shape)
