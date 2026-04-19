import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

X = np.random.randn(200, 15, 5).astype("float32")
y = np.random.randint(0, 4, size=(200,))

model = keras.Sequential([
    layers.GRU(32, input_shape=(15, 5)),
    layers.Dense(4, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=10, verbose=0)
loss, accuracy = model.evaluate(X, y, verbose=0)

print("Accuracy:", accuracy)
