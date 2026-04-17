import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def create_data(samples=600):
    numeric = np.random.rand(samples, 4).astype("float32")
    category = np.random.randint(0, 6, size=(samples, 1)).astype("int32")
    y = ((numeric[:, 0] + numeric[:, 1] > 1.0) | (category[:, 0] == 3)).astype("float32")
    return numeric, category, y


def build_model():
    numeric_input = keras.Input(shape=(4,), name="numeric")
    category_input = keras.Input(shape=(1,), dtype="int32", name="category")

    wide_part = layers.IntegerLookup(vocabulary=[0, 1, 2, 3, 4, 5], output_mode="one_hot")(category_input)

    deep_part = layers.Embedding(input_dim=7, output_dim=4)(category_input)
    deep_part = layers.Flatten()(deep_part)
    deep_part = layers.Concatenate()([numeric_input, deep_part])
    deep_part = layers.Dense(64, activation="relu")(deep_part)
    deep_part = layers.Dense(32, activation="relu")(deep_part)

    combined = layers.Concatenate()([wide_part, deep_part])
    output = layers.Dense(1, activation="sigmoid")(combined)

    return keras.Model(inputs=[numeric_input, category_input], outputs=output)


def main():
    numeric, category, y = create_data()
    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        {"numeric": numeric, "category": category},
        y,
        epochs=4,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
    )


if __name__ == "__main__":
    main()
