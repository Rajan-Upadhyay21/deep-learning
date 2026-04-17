import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def create_data(samples=500):
    city_id = np.random.randint(0, 10, size=(samples, 1)).astype("int32")
    product_id = np.random.randint(0, 15, size=(samples, 1)).astype("int32")
    numeric = np.random.rand(samples, 3).astype("float32")
    y = ((city_id[:, 0] % 2 == 0) & (numeric[:, 0] > 0.4)).astype("float32")
    return city_id, product_id, numeric, y


def build_model():
    city_input = keras.Input(shape=(1,), dtype="int32", name="city")
    product_input = keras.Input(shape=(1,), dtype="int32", name="product")
    numeric_input = keras.Input(shape=(3,), name="numeric")

    city_embed = layers.Embedding(input_dim=10, output_dim=4)(city_input)
    product_embed = layers.Embedding(input_dim=15, output_dim=5)(product_input)

    city_embed = layers.Flatten()(city_embed)
    product_embed = layers.Flatten()(product_embed)

    x = layers.Concatenate()([city_embed, product_embed, numeric_input])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs=[city_input, product_input, numeric_input], outputs=output)


def main():
    city_id, product_id, numeric, y = create_data()
    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        {"city": city_id, "product": product_id, "numeric": numeric},
        y,
        epochs=4,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
    )


if __name__ == "__main__":
    main()
