import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

products = pd.DataFrame(
    {
        "Electronics": [1, 1, 0, 0],
        "Portable": [1, 0, 0, 1],
        "Home": [0, 0, 1, 1],
    },
    index=["Laptop", "Tablet", "Vacuum", "Smart Speaker"]
)

similarity = pd.DataFrame(
    cosine_similarity(products),
    index=products.index,
    columns=products.index
)

product_name = "Laptop"
recommendations = similarity[product_name].sort_values(ascending=False)[1:]

print("Product Similarity Matrix:")
print(similarity)

print(f"\nRecommendations for {product_name}:")
print(recommendations)
