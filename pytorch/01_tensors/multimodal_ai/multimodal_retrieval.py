import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "A cat sitting on a sofa",
    "A person driving a car",
    "A dog running in a park"
]

image_embeddings = np.array([
    [0.92, 0.10, 0.20],
    [0.15, 0.93, 0.18],
    [0.20, 0.25, 0.95]
])

query_text = "dog in park"
query_embedding = np.array([[0.18, 0.20, 0.97]])

scores = cosine_similarity(query_embedding, image_embeddings)[0]
ranked_indices = np.argsort(scores)[::-1]

print("Retrieval Results:")
for idx in ranked_indices:
    print(f"{documents[idx]} -> {scores[idx]:.4f}")
