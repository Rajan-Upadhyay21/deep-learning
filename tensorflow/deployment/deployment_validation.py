import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_model():
    return keras.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])


def validate_model_file(path):
    exists = os.path.exists(path)
    print(f"Model file exists: {exists}")
    return exists


def validate_inference(model):
    sample = np.array([[0.1, 0.4, 0.8, 0.9]], dtype="float32")
    prediction = model.predict(sample, verbose=0)
    print("Inference successful. Output shape:", prediction.shape)


def main():
    x = np.random.rand(150, 4).astype("float32")
    y = (x.mean(axis=1) > 0.5).astype("float32")

    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(x, y, epochs=1, verbose=0)
    save_path = "validated_model.keras"
    model.save(save_path)

    if validate_model_file(save_path):
        loaded_model = keras.models.load_model(save_path)
        validate_inference(loaded_model)
        print("Deployment validation completed.")


if __name__ == "__main__":
    main()
