import tensorflow as tf
from tensorflow.keras import layers

from multi_head_attention import MultiHeadSelfAttention


class DecoderBlock(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

    def call(self, inputs, encoder_outputs, training=False, look_ahead_mask=None):
        attn1, _ = self.self_attention(inputs, mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.norm1(inputs + attn1)

        attn2 = self.cross_attention(query=out1, value=encoder_outputs, key=encoder_outputs)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.norm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.norm3(out2 + ffn_output)


if __name__ == "__main__":
    x = tf.random.normal((2, 10, 128))
    enc = tf.random.normal((2, 12, 128))
    block = DecoderBlock(embed_dim=128, num_heads=4, ff_dim=256)
    y = block(x, enc)
    print("Output shape:", y.shape)
