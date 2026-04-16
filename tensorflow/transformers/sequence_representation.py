import tensorflow as tf
from tensorflow.keras import layers


class SequenceRepresentation(layers.Layer):
    def __init__(self, method: str = "average"):
        super().__init__()
        if method not in {"average", "max"}:
            raise ValueError("method must be 'average' or 'max'")
        self.method = method
        self.avg_pool = layers.GlobalAveragePooling1D()
        self.max_pool = layers.GlobalMaxPooling1D()

    def call(self, inputs):
        if self.method == "average":
            return self.avg_pool(inputs)
        return self.max_pool(inputs)


if __name__ == "__main__":
    x = tf.random.normal((4, 12, 64))
    avg_layer = SequenceRepresentation(method="average")
    max_layer = SequenceRepresentation(method="max")

    print("Average pooled shape:", avg_layer(x).shape)
    print("Max pooled shape:", max_layer(x).shape)
