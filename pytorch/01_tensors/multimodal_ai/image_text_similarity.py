import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

texts = [
    "a dog playing outside",
    "a car on the road",
    "a person using a laptop"
]

text_vectors = np.array([
    [0.9, 0.1, 0.2],
    [0.1, 0.95, 0.2],
    [0.2, 0.3, 0.9]
])

image_vector = np.array([[0.88, 0.12, 0.18]])

scores = cosine_similarity(image_vector, text_vectors)[0]
best_index = np.argmax(scores)

print("Similarity Scores:")
for text, score in zip(texts, scores):
    print(f"{text} -> {score:.4f}")

print("\nBest Matching Text:")
print(texts[best_index])
