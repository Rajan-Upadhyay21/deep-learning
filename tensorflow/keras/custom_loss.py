import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

X = np.random.randn(100, 3).astype("float32")
y = np.random.randn(100, 1).astype("float32")

model = keras.Sequential([
    layers.Dense(8, activation="relu", input_shape=(3,)),
    layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss=custom_mse
)

model.fit(X, y, epochs=5, verbose=0)
print("Training completed with custom loss.")
