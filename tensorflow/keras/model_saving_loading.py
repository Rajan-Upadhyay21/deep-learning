import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

X = np.random.randn(100, 3).astype("float32")
y = np.random.randint(0, 2, size=(100,))

model = keras.Sequential([
    layers.Dense(8, activation="relu", input_shape=(3,)),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=5, verbose=0)

model.save("keras_model.keras")
print("Model saved as keras_model.keras")

loaded_model = keras.models.load_model("keras_model.keras")
print("Model loaded successfully")

predictions = loaded_model.predict(X[:5], verbose=0)
print("Predictions:")
print(predictions)
