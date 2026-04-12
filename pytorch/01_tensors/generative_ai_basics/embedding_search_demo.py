import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

items = [
    "machine learning",
    "deep learning",
    "computer vision",
    "natural language processing"
]

embeddings = np.array([
    [0.90, 0.10, 0.20],
    [0.85, 0.15, 0.30],
    [0.20, 0.95, 0.10],
    [0.25, 0.30, 0.92]
])

query = "AI language models"
query_embedding = np.array([[0.20, 0.25, 0.95]])

scores = cosine_similarity(query_embedding, embeddings)[0]
ranked_indices = np.argsort(scores)[::-1]

print("Query:")
print(query)

print("\nRanked Results:")
for index in ranked_indices:
    print(f"{items[index]} -> similarity: {scores[index]:.4f}")
