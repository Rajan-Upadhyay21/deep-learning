import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

normal_data = np.random.normal(loc=0, scale=1, size=(100, 2))
anomalies = np.array([[4, 4], [5, 5], [4.5, 5.2]])

X = np.vstack([normal_data, anomalies])

model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
labels = model.fit_predict(X)

df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
df["Label"] = labels

print(df.tail(10))
