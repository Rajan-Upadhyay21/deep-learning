import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "PyTorch is used for deep learning",
    "Transformers are used in language models",
    "RAG retrieves documents before generation"
]

document_vectors = np.array([
    [0.9, 0.1, 0.2],
    [0.2, 0.95, 0.3],
    [0.1, 0.3, 0.98]
])

query = "How does retrieval help generation?"
query_vector = np.array([[0.1, 0.25, 0.95]])

scores = cosine_similarity(query_vector, document_vectors)[0]
ranked_indices = np.argsort(scores)[::-1]

print("Query:")
print(query)

print("\nSimilarity Scores:")
for idx in ranked_indices:
    print(f"{documents[idx]} -> {scores[idx]:.4f}")
