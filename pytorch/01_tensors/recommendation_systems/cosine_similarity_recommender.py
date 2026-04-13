import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

items = pd.DataFrame(
    {
        "Feature_1": [1, 1, 0, 0],
        "Feature_2": [0, 1, 1, 0],
        "Feature_3": [0, 0, 1, 1],
    },
    index=["Item_A", "Item_B", "Item_C", "Item_D"]
)

similarity_matrix = pd.DataFrame(
    cosine_similarity(items),
    index=items.index,
    columns=items.index
)

target_item = "Item_A"
recommended_items = similarity_matrix[target_item].sort_values(ascending=False)[1:]

print("Similarity Matrix:")
print(similarity_matrix)

print(f"\nRecommendations for {target_item}:")
print(recommended_items)
