import tensorflow as tf


def create_padding_mask(sequence: tf.Tensor) -> tf.Tensor:
    mask = tf.cast(tf.math.equal(sequence, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size: int) -> tf.Tensor:
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


if __name__ == "__main__":
    sample_sequence = tf.constant([[7, 3, 0, 0], [1, 2, 5, 0]])
    padding_mask = create_padding_mask(sample_sequence)
    look_ahead_mask = create_look_ahead_mask(4)

    print("Padding mask shape:", padding_mask.shape)
    print("Look-ahead mask shape:", look_ahead_mask.shape)
