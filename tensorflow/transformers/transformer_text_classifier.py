import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from positional_encoding import PositionalEncoding
from encoder_block import EncoderBlock


def build_text_classifier(
    vocab_size: int = 15000,
    max_length: int = 120,
    embed_dim: int = 128,
    num_heads: int = 4,
    ff_dim: int = 256,
    num_classes: int = 3,
):
    inputs = keras.Input(shape=(max_length,), dtype=tf.int32)

    x = layers.Embedding(vocab_size, embed_dim)(inputs)
    x = PositionalEncoding(max_length, embed_dim)(x)
    x = EncoderBlock(embed_dim, num_heads, ff_dim)(x)
    x = EncoderBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


if __name__ == "__main__":
    model = build_text_classifier()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
