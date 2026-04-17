import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape=(96, 96, 3), num_classes=5):
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None,
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def create_dummy_data(samples=128, num_classes=5):
    x = np.random.rand(samples, 96, 96, 3).astype("float32") * 255.0
    y = np.random.randint(0, num_classes, size=(samples,))
    return x, y


def main():
    x, y = create_dummy_data()
    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x, y, epochs=2, batch_size=16, validation_split=0.2, verbose=1)


if __name__ == "__main__":
    main()
