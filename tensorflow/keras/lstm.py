import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

X = np.random.randn(200, 12, 6).astype("float32")
y = np.random.randint(0, 2, size=(200,))

model = keras.Sequential([
    layers.LSTM(32, input_shape=(12, 6)),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=10, verbose=0)
loss, accuracy = model.evaluate(X, y, verbose=0)

print("Accuracy:", accuracy)
