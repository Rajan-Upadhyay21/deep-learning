import tensorflow as tf
from tensorflow.keras import layers


class TokenEmbedding(layers.Layer):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)

    def call(self, inputs):
        return self.embedding(inputs)


if __name__ == "__main__":
    tokens = tf.constant([[1, 5, 9, 3], [2, 8, 4, 7]])
    layer = TokenEmbedding(vocab_size=10000, embed_dim=64)
    embeddings = layer(tokens)
    print("Embeddings shape:", embeddings.shape)
