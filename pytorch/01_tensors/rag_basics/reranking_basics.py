documents = [
    {"text": "RAG retrieves documents before generation.", "initial_score": 0.82, "keyword_overlap": 2},
    {"text": "Transformers use self-attention mechanisms.", "initial_score": 0.79, "keyword_overlap": 0},
    {"text": "RAG improves grounding with retrieved context.", "initial_score": 0.80, "keyword_overlap": 3},
]

for doc in documents:
    doc["reranked_score"] = doc["initial_score"] + 0.05 * doc["keyword_overlap"]

reranked = sorted(documents, key=lambda x: x["reranked_score"], reverse=True)

print("Reranked Results:")
for doc in reranked:
    print(doc)
