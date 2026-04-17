import tensorflow as tf
from tensorflow.keras.layers import HashedCrossing


def main():
    cities = tf.constant([["Chicago"], ["New York"], ["Chicago"], ["Austin"]])
    devices = tf.constant([["Mobile"], ["Desktop"], ["Tablet"], ["Mobile"]])

    crossing = HashedCrossing(num_bins=8)
    crossed_features = crossing((cities, devices))

    print("Crossed feature bins:\n", crossed_features.numpy())


if __name__ == "__main__":
    main()
