import pandas as pd

implicit_feedback = pd.DataFrame(
    {
        "Item_A": [1, 1, 0, 0],
        "Item_B": [1, 0, 1, 0],
        "Item_C": [0, 1, 1, 1],
        "Item_D": [0, 0, 1, 1],
    },
    index=["User1", "User2", "User3", "User4"]
)

user_preferences = implicit_feedback.sum(axis=1)
item_popularity = implicit_feedback.sum(axis=0)

print("Implicit Feedback Matrix:")
print(implicit_feedback)

print("\nUser Activity Levels:")
print(user_preferences)

print("\nItem Popularity:")
print(item_popularity)
