import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "PyTorch is used for deep learning and neural networks.",
    "Transformers are important in modern language models.",
    "Retrieval augmented generation combines retrieval with generation."
]

document_embeddings = np.array([
    [0.9, 0.2, 0.1],
    [0.2, 0.95, 0.3],
    [0.1, 0.4, 0.98]
])

query = "How does retrieval improve AI responses?"
query_embedding = np.array([[0.1, 0.35, 0.95]])

similarities = cosine_similarity(query_embedding, document_embeddings)[0]
best_doc_index = np.argmax(similarities)
retrieved_context = documents[best_doc_index]

response = f"Using retrieved context: {retrieved_context}"

print("Query:")
print(query)

print("\nRetrieved Context:")
print(retrieved_context)

print("\nGenerated Response:")
print(response)
