import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def create_customer_data(samples=900):
    tenure = np.random.rand(samples, 1).astype("float32")
    monthly_spend = np.random.rand(samples, 1).astype("float32")
    support_calls = np.random.randint(0, 6, size=(samples, 1)).astype("float32")
    contract_type = np.random.randint(0, 3, size=(samples, 1)).astype("int32")

    x_numeric = np.concatenate([tenure, monthly_spend, support_calls], axis=1)
    churn = ((tenure[:, 0] < 0.3) & (support_calls[:, 0] > 2)) | (contract_type[:, 0] == 0)
    y = churn.astype("float32")
    return x_numeric, contract_type, y


def build_model():
    numeric_input = keras.Input(shape=(3,), name="numeric")
    contract_input = keras.Input(shape=(1,), dtype="int32", name="contract")

    contract_embed = layers.Embedding(input_dim=3, output_dim=3)(contract_input)
    contract_embed = layers.Flatten()(contract_embed)

    x = layers.Concatenate()([numeric_input, contract_embed])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs=[numeric_input, contract_input], outputs=output)


def main():
    x_numeric, contract_type, y = create_customer_data()
    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        {"numeric": x_numeric, "contract": contract_type},
        y,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
    )


if __name__ == "__main__":
    main()
