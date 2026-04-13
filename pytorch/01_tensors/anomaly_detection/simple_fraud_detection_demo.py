import pandas as pd
from sklearn.ensemble import IsolationForest

transactions = pd.DataFrame({
    "amount": [50, 60, 55, 48, 52, 5000, 47, 53, 49, 5200],
    "time_gap": [5, 4, 6, 5, 4, 1, 5, 6, 5, 1]
})

model = IsolationForest(contamination=0.2, random_state=42)
transactions["fraud_label"] = model.fit_predict(transactions)

print(transactions)
