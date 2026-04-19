import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

X = np.random.randn(200, 4).astype("float32")
y = np.random.randint(0, 3, size=(200,))

model = keras.Sequential([
    layers.Dense(16, activation="relu", input_shape=(4,)),
    layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=2,
    restore_best_weights=True
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="best_keras_model.keras",
    monitor="loss",
    save_best_only=True
)

model.fit(
    X,
    y,
    epochs=10,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)
