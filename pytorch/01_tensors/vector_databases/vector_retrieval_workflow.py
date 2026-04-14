import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

store = [
    {"id": "c1", "text": "Transformers use attention.", "embedding": np.array([0.9, 0.1, 0.2])},
    {"id": "c2", "text": "RAG retrieves external knowledge.", "embedding": np.array([0.1, 0.4, 0.95])},
    {"id": "c3", "text": "CNNs are strong for image data.", "embedding": np.array([0.8, 0.2, 0.1])},
]

query = "What does RAG do?"
query_embedding = np.array([[0.1, 0.35, 0.97]])

embeddings = np.array([item["embedding"] for item in store])
scores = cosine_similarity(query_embedding, embeddings)[0]
best_index = int(np.argmax(scores))

print("Query:")
print(query)

print("\nBest Retrieved Result:")
print(store[best_index]["text"])

print("\nSimilarity Score:")
print(scores[best_index])
