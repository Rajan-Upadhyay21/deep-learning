import tensorflow as tf
import numpy as np

X = np.random.randn(200, 4).astype(np.float32)
y = np.random.randint(0, 3, size=(200,))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(4,)),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=2,
    restore_best_weights=True
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_tf_model.keras",
    save_best_only=True,
    monitor="loss"
)

model.fit(
    X,
    y,
    epochs=10,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

print("Training complete with callbacks.")
