import tensorflow as tf
from tensorflow.keras import layers


class ContextFusion(layers.Layer):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.projection = layers.Dense(embed_dim, activation="relu")

    def call(self, attention_output, residual_input):
        fused = attention_output + residual_input
        return self.projection(fused)


if __name__ == "__main__":
    attention_output = tf.random.normal((2, 12, 64))
    residual_input = tf.random.normal((2, 12, 64))

    layer = ContextFusion(embed_dim=64)
    output = layer(attention_output, residual_input)
    print("Output shape:", output.shape)
