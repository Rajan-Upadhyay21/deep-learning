import tensorflow as tf


def compute_attention_scores(query: tf.Tensor, key: tf.Tensor) -> tf.Tensor:
    scores = tf.matmul(query, key, transpose_b=True)
    scale = tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))
    return scores / scale


if __name__ == "__main__":
    q = tf.random.normal((2, 10, 64))
    k = tf.random.normal((2, 10, 64))
    scores = compute_attention_scores(q, k)
    print("Attention scores shape:", scores.shape)
