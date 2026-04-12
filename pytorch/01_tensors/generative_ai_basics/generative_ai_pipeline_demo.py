from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def tokenize(text):
    return text.lower().replace(".", "").split()

def retrieve(query_embedding, document_embeddings, documents):
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    best_index = int(np.argmax(similarities))
    return documents[best_index], similarities

def generate_answer(prompt, context):
    return f"Prompt: {prompt}\nContext used: {context}\nGenerated answer: This is a mock generative AI response."

documents = [
    "LLMs generate text based on context and probability distributions.",
    "Embeddings help semantic search and retrieval.",
    "Prompt engineering helps improve AI outputs."
]

document_embeddings = np.array([
    [0.9, 0.1, 0.2],
    [0.2, 0.95, 0.4],
    [0.4, 0.2, 0.9]
])

user_query = "How do embeddings help AI systems?"
query_tokens = tokenize(user_query)
query_embedding = np.array([0.2, 0.9, 0.35])

retrieved_doc, similarity_scores = retrieve(query_embedding, document_embeddings, documents)
answer = generate_answer(user_query, retrieved_doc)

print("User Query:")
print(user_query)

print("\nTokens:")
print(query_tokens)

print("\nRetrieved Document:")
print(retrieved_doc)

print("\nSimilarity Scores:")
print(similarity_scores)

print("\nFinal Answer:")
print(answer)
