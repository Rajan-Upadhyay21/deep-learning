import pandas as pd
from sklearn.linear_model import LinearRegression

train_data = pd.DataFrame({
    "feature": [1, 2, 3, 4, 5],
    "target": [2, 4, 6, 8, 10]
})

batch_data = pd.DataFrame({
    "feature": [6, 7, 8, 9]
})

model = LinearRegression()
model.fit(train_data[["feature"]], train_data["target"])

batch_data["prediction"] = model.predict(batch_data[["feature"]])

print("Batch Prediction Results:")
print(batch_data)
