import tensorflow as tf
from tensorflow.keras.layers import StringLookup, CategoryEncoding


def main():
    categories = tf.constant([["red"], ["blue"], ["green"], ["red"], ["blue"]])

    lookup = StringLookup(output_mode="int")
    lookup.adapt(categories)

    encoded_ids = lookup(categories)

    one_hot = CategoryEncoding(num_tokens=lookup.vocabulary_size(), output_mode="one_hot")
    one_hot_vectors = one_hot(encoded_ids)

    print("Encoded IDs:\n", encoded_ids.numpy())
    print("One-hot vectors:\n", one_hot_vectors.numpy())


if __name__ == "__main__":
    main()
