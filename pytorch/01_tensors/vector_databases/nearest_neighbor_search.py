import numpy as np
from sklearn.neighbors import NearestNeighbors

vectors = np.array([
    [0.90, 0.10, 0.20],
    [0.15, 0.85, 0.25],
    [0.20, 0.25, 0.95],
    [0.88, 0.12, 0.22]
])

labels = ["item_1", "item_2", "item_3", "item_4"]
query = np.array([[0.89, 0.11, 0.21]])

nn = NearestNeighbors(n_neighbors=2, metric="cosine")
nn.fit(vectors)

distances, indices = nn.kneighbors(query)

print("Nearest Neighbor Results:")
for distance, index in zip(distances[0], indices[0]):
    print(f"{labels[index]} -> distance: {distance:.4f}")
