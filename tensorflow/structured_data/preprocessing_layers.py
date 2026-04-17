import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def create_data(samples=500):
    numeric_feature = np.random.rand(samples, 1).astype("float32")
    category_ids = np.random.randint(0, 5, size=(samples, 1)).astype("int32")
    y = ((numeric_feature[:, 0] > 0.5) | (category_ids[:, 0] == 2)).astype("float32")
    return numeric_feature, category_ids, y


def build_model():
    numeric_input = keras.Input(shape=(1,), name="numeric")
    category_input = keras.Input(shape=(1,), dtype="int32", name="category")

    normalizer = layers.Normalization()
    lookup = layers.IntegerLookup(vocabulary=[0, 1, 2, 3, 4], output_mode="one_hot")

    x_num = normalizer(numeric_input)
    x_cat = lookup(category_input)

    x = layers.Concatenate()([x_num, x_cat])
    x = layers.Dense(32, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=[numeric_input, category_input], outputs=output)
    return model, normalizer


def main():
    numeric_feature, category_ids, y = create_data()
    model, normalizer = build_model()
    normalizer.adapt(numeric_feature)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        {"numeric": numeric_feature, "category": category_ids},
        y,
        epochs=4,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
    )


if __name__ == "__main__":
    main()
