import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

normal_data = np.random.normal(loc=50, scale=5, size=(100, 2))
anomalies = np.array([[90, 95], [100, 110], [85, 92]])

X = np.vstack([normal_data, anomalies])

model = IsolationForest(contamination=0.03, random_state=42)
labels = model.fit_predict(X)

df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
df["Label"] = labels

print(df.tail(10))
