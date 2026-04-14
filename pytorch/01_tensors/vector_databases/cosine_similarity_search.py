import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

documents = ["doc_1", "doc_2", "doc_3"]

document_vectors = np.array([
    [0.90, 0.10, 0.20],
    [0.15, 0.85, 0.25],
    [0.20, 0.25, 0.95]
])

query_vector = np.array([[0.18, 0.22, 0.97]])

scores = cosine_similarity(query_vector, document_vectors)[0]
ranked_indices = np.argsort(scores)[::-1]

print("Cosine Similarity Search Results:")
for index in ranked_indices:
    print(f"{documents[index]} -> {scores[index]:.4f}")
