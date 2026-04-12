import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "PyTorch is a deep learning framework.",
    "Transformers are used in modern language models.",
    "CNNs are useful for image classification."
]

doc_embeddings = np.array([
    [0.9, 0.1, 0.3],
    [0.2, 0.95, 0.4],
    [0.3, 0.2, 0.98]
])

query = "What are transformers used for?"
query_embedding = np.array([[0.1, 0.9, 0.3]])

similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
best_index = np.argmax(similarities)

print("Query:")
print(query)

print("\nRetrieved Document:")
print(documents[best_index])

print("\nSimilarity Scores:")
print(similarities)
