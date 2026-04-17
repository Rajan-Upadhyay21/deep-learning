import numpy as np
import tensorflow as tf


def create_data(samples=100, features=6):
    x = np.random.rand(samples, features).astype("float32")
    y = (x.sum(axis=1) > 3.0).astype("float32")
    return x, y


def create_dataset(x, y, batch_size=16):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(len(x)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def main():
    x, y = create_data()
    dataset = create_dataset(x, y)

    for features, labels in dataset.take(2):
        print("Feature batch shape:", features.shape)
        print("Label batch shape:", labels.shape)


if __name__ == "__main__":
    main()
