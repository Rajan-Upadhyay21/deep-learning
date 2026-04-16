import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_activation_demo_model(activation: str) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(20,)),
        layers.Dense(64, activation=activation),
        layers.Dense(32, activation=activation),
        layers.Dense(1, activation="sigmoid")
    ])
    return model


if __name__ == "__main__":
    activations = ["relu", "sigmoid", "tanh", "elu", "selu"]

    for act in activations:
        model = build_activation_demo_model(act)
        model.summary()
        print(f"\nBuilt model using activation: {act}\n")
