import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_data():
    x = np.random.rand(600, 10).astype("float32")
    y = (x.sum(axis=1) > 5).astype("float32")
    return x, y


def build_model(units=32, learning_rate=1e-3):
    model = keras.Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(units, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    x, y = create_data()

    search_space = [
        {"units": 16, "learning_rate": 1e-3},
        {"units": 32, "learning_rate": 1e-3},
        {"units": 64, "learning_rate": 5e-4},
    ]

    best_config = None
    best_val_acc = -1.0

    for config in search_space:
        model = build_model(**config)
        history = model.fit(
            x,
            y,
            validation_split=0.2,
            epochs=4,
            batch_size=32,
            verbose=0
        )
        val_acc = history.history["val_accuracy"][-1]
        print(f"Config {config} -> val_accuracy={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = config

    print("Best configuration:", best_config)
    print("Best validation accuracy:", round(best_val_acc, 4))


if __name__ == "__main__":
    main()
