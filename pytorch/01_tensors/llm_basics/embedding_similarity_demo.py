import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embedding_1 = np.array([[0.2, 0.8, 0.5, 0.1]])
embedding_2 = np.array([[0.1, 0.7, 0.45, 0.2]])
embedding_3 = np.array([[0.9, 0.1, 0.2, 0.8]])

sim_1_2 = cosine_similarity(embedding_1, embedding_2)[0][0]
sim_1_3 = cosine_similarity(embedding_1, embedding_3)[0][0]

print("Similarity between embedding_1 and embedding_2:")
print(sim_1_2)

print("\nSimilarity between embedding_1 and embedding_3:")
print(sim_1_3)
