import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

text_embedding = np.array([[0.9, 0.1, 0.2, 0.3]])
image_embedding = np.array([[0.85, 0.12, 0.25, 0.28]])
other_image_embedding = np.array([[0.2, 0.8, 0.3, 0.9]])

similarity_1 = cosine_similarity(text_embedding, image_embedding)[0][0]
similarity_2 = cosine_similarity(text_embedding, other_image_embedding)[0][0]

print("Similarity between text and matching image:")
print(similarity_1)

print("\nSimilarity between text and different image:")
print(similarity_2)
