import tensorflow as tf
from tensorflow.keras import layers


class ClassificationHead(layers.Layer):
    def __init__(self, hidden_dim: int, num_classes: int, dropout_rate: float = 0.2):
        super().__init__()
        self.pool = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(dropout_rate)
        self.dense = layers.Dense(hidden_dim, activation="relu")
        self.output_layer = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.pool(inputs)
        x = self.dropout(x, training=training)
        x = self.dense(x)
        return self.output_layer(x)


if __name__ == "__main__":
    x = tf.random.normal((4, 20, 128))
    head = ClassificationHead(hidden_dim=64, num_classes=3)
    y = head(x)
    print("Output shape:", y.shape)
