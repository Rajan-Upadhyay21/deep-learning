import tensorflow as tf
import numpy as np

X = np.random.randn(100, 3).astype(np.float32)
y = np.random.randint(0, 2, size=(100,))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation="relu", input_shape=(3,)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=5, verbose=0)

model.save("saved_tf_model.keras")
print("Model saved as saved_tf_model.keras")

loaded_model = tf.keras.models.load_model("saved_tf_model.keras")
print("Model loaded successfully")

predictions = loaded_model.predict(X[:5], verbose=0)
print("Predictions:")
print(predictions)
