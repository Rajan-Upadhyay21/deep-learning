retrieved_docs = [
    "RAG combines retrieval and generation.",
    "Transformers are useful for NLP.",
    "Embeddings help semantic similarity."
]

relevant_docs = {
    "RAG combines retrieval and generation.",
    "Embeddings help semantic similarity."
}

hits = len(set(retrieved_docs) & relevant_docs)

precision_at_k = hits / len(retrieved_docs)
recall_at_k = hits / len(relevant_docs)

print("Retrieved Docs:")
print(retrieved_docs)

print("\nRelevant Docs:")
print(relevant_docs)

print("\nPrecision@K:", precision_at_k)
print("Recall@K:", recall_at_k)
