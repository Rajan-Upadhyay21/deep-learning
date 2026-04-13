import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.DataFrame(
    {
        "Movie_A": [5, 4, 0, 1],
        "Movie_B": [4, 0, 0, 1],
        "Movie_C": [1, 1, 0, 5],
        "Movie_D": [0, 0, 5, 4],
    },
    index=["User1", "User2", "User3", "User4"]
)

item_similarity = pd.DataFrame(
    cosine_similarity(ratings.T),
    index=ratings.columns,
    columns=ratings.columns
)

print("Ratings Matrix:")
print(ratings)

print("\nItem Similarity Matrix:")
print(item_similarity)
