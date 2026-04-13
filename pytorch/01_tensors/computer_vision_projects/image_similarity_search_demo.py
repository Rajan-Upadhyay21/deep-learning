import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

image_features = np.array([
    [0.9, 0.1, 0.2],
    [0.85, 0.15, 0.25],
    [0.1, 0.9, 0.3],
    [0.2, 0.8, 0.4]
])

image_names = ["image_1.jpg", "image_2.jpg", "image_3.jpg", "image_4.jpg"]

query_feature = np.array([[0.88, 0.12, 0.22]])

scores = cosine_similarity(query_feature, image_features)[0]
ranked_indices = np.argsort(scores)[::-1]

print("Similarity Search Results:")
for index in ranked_indices:
    print(f"{image_names[index]} -> similarity: {scores[index]:.4f}")
