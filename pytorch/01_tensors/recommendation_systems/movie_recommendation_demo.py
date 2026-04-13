import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

movie_features = pd.DataFrame(
    {
        "Action": [1, 1, 0, 0, 1],
        "Comedy": [0, 1, 1, 0, 0],
        "Drama": [0, 0, 1, 1, 1],
    },
    index=["Inception", "Rush Hour", "The Mask", "Titanic", "The Dark Knight"]
)

similarity = pd.DataFrame(
    cosine_similarity(movie_features),
    index=movie_features.index,
    columns=movie_features.index
)

movie_name = "Inception"
recommendations = similarity[movie_name].sort_values(ascending=False)[1:]

print("Movie Similarity Matrix:")
print(similarity)

print(f"\nRecommendations for {movie_name}:")
print(recommendations)
