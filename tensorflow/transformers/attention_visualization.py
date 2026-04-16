import tensorflow as tf
import matplotlib.pyplot as plt

from self_attention import SelfAttention


def visualize_attention(sequence_length=8, embed_dim=32):
    x = tf.random.normal((1, sequence_length, embed_dim))
    attention_layer = SelfAttention(embed_dim=embed_dim)
    _, attention_weights = attention_layer(x)

    attention_map = attention_weights[0].numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(attention_map, aspect="auto")
    plt.title("Self-Attention Weights")
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_attention()
