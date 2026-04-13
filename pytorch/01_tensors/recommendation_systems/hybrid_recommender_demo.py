import pandas as pd

collaborative_scores = pd.Series(
    [0.90, 0.75, 0.60, 0.40],
    index=["Item_A", "Item_B", "Item_C", "Item_D"]
)

content_scores = pd.Series(
    [0.85, 0.80, 0.55, 0.50],
    index=["Item_A", "Item_B", "Item_C", "Item_D"]
)

hybrid_scores = 0.5 * collaborative_scores + 0.5 * content_scores
recommendations = hybrid_scores.sort_values(ascending=False)

print("Collaborative Scores:")
print(collaborative_scores)

print("\nContent-Based Scores:")
print(content_scores)

print("\nHybrid Recommendation Scores:")
print(recommendations)
