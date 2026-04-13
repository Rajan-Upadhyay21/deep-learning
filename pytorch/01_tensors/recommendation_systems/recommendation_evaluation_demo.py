relevant_items = {"Item_A", "Item_C", "Item_D"}
recommended_items = ["Item_A", "Item_B", "Item_C"]

hits = len(set(recommended_items) & relevant_items)

precision_at_k = hits / len(recommended_items)
recall_at_k = hits / len(relevant_items)

print("Relevant Items:", relevant_items)
print("Recommended Items:", recommended_items)
print("Precision@K:", precision_at_k)
print("Recall@K:", recall_at_k)
