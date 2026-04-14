import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "PyTorch is a deep learning framework.",
    "RAG helps connect LLMs to external knowledge.",
    "Vector search retrieves semantically similar items."
]

document_vectors = np.array([
    [0.90, 0.10, 0.20],
    [0.10, 0.40, 0.95],
    [0.20, 0.25, 0.88]
])

def encode_query(query):
    if "rag" in query.lower():
        return np.array([0.12, 0.38, 0.96])
    return np.array([0.20, 0.20, 0.20])

def search(query, docs, vectors):
    query_vector = encode_query(query).reshape(1, -1)
    scores = cosine_similarity(query_vector, vectors)[0]
    ranked = np.argsort(scores)[::-1]
    return [(docs[i], scores[i]) for i in ranked]

query = "How does RAG help AI?"
results = search(query, documents, document_vectors)

print("Semantic Search Results:")
for doc, score in results:
    print(f"{doc} -> {score:.4f}")
