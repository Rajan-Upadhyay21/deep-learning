import tensorflow as tf
import numpy as np


def preprocess_text_sequences(text_sequences, max_length=100):
    padded = []
    for seq in text_sequences:
        seq = seq[:max_length]
        seq = seq + [0] * (max_length - len(seq))
        padded.append(seq)
    return np.array(padded, dtype=np.int32)


def predict_sentiment(model_path: str, tokenized_samples):
    model = tf.keras.models.load_model(model_path, compile=False)
    processed = preprocess_text_sequences(tokenized_samples)
    predictions = model.predict(processed, verbose=0)

    for i, prediction in enumerate(predictions, start=1):
        label = "Positive" if prediction[0] >= 0.5 else "Negative"
        print(f"Sample {i}: {label} ({prediction[0]:.4f})")


if __name__ == "__main__":
    sample_inputs = [
        [12, 45, 87, 34, 90],
        [8, 3, 2, 1],
    ]
    predict_sentiment("sentiment_transformer_model.keras", sample_inputs)
