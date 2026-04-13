import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "PyTorch supports neural network training.",
    "RAG retrieves external context for answers.",
    "CNNs are useful for image classification.",
    "Transformers power modern language models."
]

document_vectors = np.array([
    [0.9, 0.1, 0.2],
    [0.1, 0.3, 0.95],
    [0.8, 0.2, 0.1],
    [0.2, 0.95, 0.3]
])

query = "What is RAG?"
query_vector = np.array([[0.1, 0.25, 0.98]])

scores = cosine_similarity(query_vector, document_vectors)[0]
top_k = 2
top_indices = np.argsort(scores)[::-1][:top_k]

retrieved_docs = [documents[i] for i in top_indices]

print("Retrieved Documents:")
for doc in retrieved_docs:
    print(doc)
