import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def encode_text(text):
    if "dog" in text.lower():
        return np.array([0.2, 0.1, 0.95])
    if "car" in text.lower():
        return np.array([0.1, 0.95, 0.2])
    return np.array([0.8, 0.2, 0.1])

def retrieve_best_match(query_vector, image_vectors, labels):
    scores = cosine_similarity([query_vector], image_vectors)[0]
    best_index = int(np.argmax(scores))
    return labels[best_index], scores

labels = ["cat image", "car image", "dog image"]
image_vectors = np.array([
    [0.85, 0.12, 0.10],
    [0.12, 0.90, 0.25],
    [0.18, 0.15, 0.96]
])

query = "Find the dog picture"
query_vector = encode_text(query)
best_label, scores = retrieve_best_match(query_vector, image_vectors, labels)

print("Query:")
print(query)

print("\nQuery Vector:")
print(query_vector)

print("\nSimilarity Scores:")
print(scores)

print("\nBest Match:")
print(best_label)
