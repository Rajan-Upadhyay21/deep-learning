import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class BasicVectorDatabase:
    def __init__(self):
        self.records = []

    def add(self, record_id, text, embedding, metadata=None):
        self.records.append({
            "id": record_id,
            "text": text,
            "embedding": np.array(embedding),
            "metadata": metadata or {}
        })

    def search(self, query_embedding, top_k=2):
        embeddings = np.array([record["embedding"] for record in self.records])
        scores = cosine_similarity([query_embedding], embeddings)[0]
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            record = self.records[idx]
            results.append({
                "id": record["id"],
                "text": record["text"],
                "score": float(scores[idx]),
                "metadata": record["metadata"]
            })
        return results

db = BasicVectorDatabase()
db.add("1", "PyTorch is used for deep learning.", [0.9, 0.1, 0.2], {"topic": "dl"})
db.add("2", "RAG combines retrieval and generation.", [0.1, 0.4, 0.95], {"topic": "llm"})
db.add("3", "Vector databases support semantic search.", [0.2, 0.3, 0.9], {"topic": "retrieval"})

query_embedding = np.array([0.15, 0.35, 0.96])
results = db.search(query_embedding, top_k=2)

print("Vector Database Search Results:")
for result in results:
    print(result)
