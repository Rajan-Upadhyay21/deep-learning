import tensorflow as tf
from tensorflow.keras import layers


class SelfAttention(layers.Layer):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = layers.Dense(embed_dim)
        self.key = layers.Dense(embed_dim)
        self.value = layers.Dense(embed_dim)

    def call(self, inputs):
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        scores = tf.matmul(q, k, transpose_b=True)
        scale = tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))
        attention_weights = tf.nn.softmax(scores / scale, axis=-1)
        return tf.matmul(attention_weights, v), attention_weights


if __name__ == "__main__":
    x = tf.random.normal((2, 12, 64))
    layer = SelfAttention(embed_dim=64)
    output, weights = layer(x)
    print("Output shape:", output.shape)
    print("Weights shape:", weights.shape)
