import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "PyTorch is a deep learning library.",
    "Transformers are widely used in LLMs.",
    "RAG combines retrieval and generation for grounded answers."
]

document_vectors = np.array([
    [0.9, 0.1, 0.2],
    [0.2, 0.95, 0.3],
    [0.1, 0.4, 0.98]
])

def retrieve(query_vector, docs, doc_vectors, top_k=1):
    scores = cosine_similarity([query_vector], doc_vectors)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [docs[i] for i in top_indices], scores

def generate_answer(query, context):
    return f"Question: {query}\nContext: {context}\nAnswer: This is a grounded response based on retrieved context."

user_query = "What is RAG in AI?"
query_vector = np.array([0.1, 0.35, 0.97])

retrieved_docs, scores = retrieve(query_vector, documents, document_vectors, top_k=1)
answer = generate_answer(user_query, retrieved_docs[0])

print("Retrieved Document:")
print(retrieved_docs[0])

print("\nAnswer:")
print(answer)
