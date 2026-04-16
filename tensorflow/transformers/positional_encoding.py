import tensorflow as tf
from tensorflow.keras import layers


class PositionalEncoding(layers.Layer):
    def __init__(self, max_length: int, embed_dim: int):
        super().__init__()
        self.pos_encoding = self._build_positional_encoding(max_length, embed_dim)

    def _build_positional_encoding(self, max_length: int, embed_dim: int) -> tf.Tensor:
        positions = tf.range(max_length, dtype=tf.float32)[:, tf.newaxis]
        dimensions = tf.range(embed_dim, dtype=tf.float32)[tf.newaxis, :]

        angle_rates = 1.0 / tf.pow(
            10000.0, (2 * (dimensions // 2)) / tf.cast(embed_dim, tf.float32)
        )
        angle_rads = positions * angle_rates

        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]


if __name__ == "__main__":
    x = tf.random.normal((2, 10, 64))
    layer = PositionalEncoding(max_length=100, embed_dim=64)
    y = layer(x)
    print("Output shape:", y.shape)
