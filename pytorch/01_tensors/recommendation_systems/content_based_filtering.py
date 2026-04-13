import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

items = pd.DataFrame(
    {
        "Action": [1, 1, 0, 0],
        "Comedy": [0, 1, 1, 0],
        "Drama": [0, 0, 1, 1],
    },
    index=["Movie_A", "Movie_B", "Movie_C", "Movie_D"]
)

similarity = pd.DataFrame(
    cosine_similarity(items),
    index=items.index,
    columns=items.index
)

print("Item Feature Matrix:")
print(items)

print("\nContent Similarity Matrix:")
print(similarity)
