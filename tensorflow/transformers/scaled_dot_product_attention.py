import tensorflow as tf


def scaled_dot_product_attention(query, key, value, mask=None):
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    if mask is not None:
        logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output, attention_weights


if __name__ == "__main__":
    q = tf.random.normal((2, 8, 32))
    k = tf.random.normal((2, 8, 32))
    v = tf.random.normal((2, 8, 32))

    output, weights = scaled_dot_product_attention(q, k, v)
    print("Output shape:", output.shape)
    print("Weights shape:", weights.shape)
