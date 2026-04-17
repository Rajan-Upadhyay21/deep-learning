import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def create_transaction_data(samples=1200, features=12):
    x = np.random.rand(samples, features).astype("float32")
    risk_score = 2.0 * x[:, 0] + 1.5 * x[:, 3] + 1.2 * x[:, 7]
    y = (risk_score > 3.2).astype("float32")
    return x, y


def build_model(input_dim=12):
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])


def main():
    x, y = create_transaction_data()
    model = build_model(x.shape[1])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name="auc")])
    model.fit(x, y, epochs=5, batch_size=32, validation_split=0.2, verbose=1)


if __name__ == "__main__":
    main()
