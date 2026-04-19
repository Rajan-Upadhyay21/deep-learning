from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(4,))
x = layers.Dense(16, activation="relu")(inputs)
x = layers.Dense(8, activation="relu")(x)
outputs = layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()
