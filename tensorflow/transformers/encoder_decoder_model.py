import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from positional_encoding import PositionalEncoding
from encoder_block import EncoderBlock
from decoder_block import DecoderBlock


class EncoderDecoderTransformer(keras.Model):
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
    ):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.position = PositionalEncoding(max_length, embed_dim)

        self.encoder = EncoderBlock(embed_dim, num_heads, ff_dim)
        self.decoder = DecoderBlock(embed_dim, num_heads, ff_dim)

        self.output_layer = layers.Dense(vocab_size, activation="softmax")

    def call(self, encoder_inputs, decoder_inputs, training=False):
        enc_x = self.embedding(encoder_inputs)
        enc_x = self.position(enc_x)
        enc_out = self.encoder(enc_x, training=training)

        dec_x = self.embedding(decoder_inputs)
        dec_x = self.position(dec_x)
        dec_out = self.decoder(dec_x, enc_out, training=training)

        return self.output_layer(dec_out)


if __name__ == "__main__":
    enc_inputs = tf.random.uniform((2, 15), maxval=1000, dtype=tf.int32)
    dec_inputs = tf.random.uniform((2, 12), maxval=1000, dtype=tf.int32)

    model = EncoderDecoderTransformer(vocab_size=1000, max_length=50)
    outputs = model(enc_inputs, dec_inputs)
    print("Output shape:", outputs.shape)
