import numpy as np
import tensorflow as tf


def create_series(length=100):
    time = np.arange(length, dtype="float32")
    return np.sin(0.2 * time).astype("float32")


def create_windowed_dataset(series, window_size=10, batch_size=4):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def main():
    series = create_series()
    dataset = create_windowed_dataset(series, window_size=8, batch_size=2)

    for inputs, targets in dataset.take(2):
        print("Input batch shape:", inputs.shape)
        print("Target batch shape:", targets.shape)


if __name__ == "__main__":
    main()
