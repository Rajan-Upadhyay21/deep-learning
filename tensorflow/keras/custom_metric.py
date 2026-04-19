import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def custom_accuracy(y_true, y_pred):
    y_pred_labels = tf.argmax(y_pred, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(tf.cast(y_true, tf.int64), y_pred_labels), tf.float32))

X = np.random.randn(150, 4).astype("float32")
y = np.random.randint(0, 3, size=(150,))

model = keras.Sequential([
    layers.Dense(16, activation="relu", input_shape=(4,)),
    layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[custom_accuracy]
)

model.fit(X, y, epochs=5, verbose=0)
print("Training completed with custom metric.")
