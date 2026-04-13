import numpy as np
from sklearn.ensemble import IsolationForest

normal_data = np.random.normal(loc=0, scale=1, size=(100, 2))
anomalies = np.array([[4, 4], [5, 5], [4.5, 5.2]])

X = np.vstack([normal_data, anomalies])

model = IsolationForest(contamination=0.03, random_state=42)
model.fit(X)

scores = model.decision_function(X)

print("Anomaly Scores:")
print(scores[-10:])
