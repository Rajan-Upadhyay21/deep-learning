import tensorflow as tf
from tensorflow.keras import datasets, layers, models

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

X_train = X_train[..., tf.newaxis] / 255.0
X_test = X_test[..., tf.newaxis] / 255.0

model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=2, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Test Accuracy:")
print(accuracy)
