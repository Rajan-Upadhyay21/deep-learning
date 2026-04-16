import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from positional_encoding import PositionalEncoding
from encoder_block import EncoderBlock


class TransformerClassifier(keras.Model):
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.position = PositionalEncoding(max_length, embed_dim)
        self.dropout = layers.Dropout(dropout_rate)
        self.encoder_layers = [
            EncoderBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        self.pool = layers.GlobalAveragePooling1D()
        self.classifier = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False, mask=None):
        x = self.embedding(inputs)
        x = self.position(x)
        x = self.dropout(x, training=training)

        for encoder in self.encoder_layers:
            x = encoder(x, training=training, mask=mask)

        x = self.pool(x)
        return self.classifier(x)


if __name__ == "__main__":
    sample_inputs = tf.random.uniform((8, 40), minval=0, maxval=5000, dtype=tf.int32)
    model = TransformerClassifier(vocab_size=5000, max_length=40)
    outputs = model(sample_inputs)
    print("Output shape:", outputs.shape)
