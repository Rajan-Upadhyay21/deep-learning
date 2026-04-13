
Code files:

## `user_based_collaborative_filtering.py`

```python
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

user_similarity = pd.DataFrame(
    cosine_similarity(ratings),
    index=ratings.index,
    columns=ratings.index
)

print("Ratings Matrix:")
print(ratings)

print("\nUser Similarity Matrix:")
print(user_similarity)
