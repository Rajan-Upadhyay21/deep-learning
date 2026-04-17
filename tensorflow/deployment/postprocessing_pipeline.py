import numpy as np


def convert_scores_to_labels(scores, threshold=0.5):
    scores = np.array(scores, dtype="float32")
    labels = ["Class 1" if score >= threshold else "Class 0" for score in scores]
    return labels


def attach_confidence(scores):
    return [f"{score:.2%}" for score in scores]


def main():
    raw_scores = [0.12, 0.78, 0.56, 0.32]
    labels = convert_scores_to_labels(raw_scores, threshold=0.5)
    confidences = attach_confidence(raw_scores)

    for idx, (label, confidence) in enumerate(zip(labels, confidences), start=1):
        print(f"Sample {idx}: {label}, confidence={confidence}")


if __name__ == "__main__":
    main()
