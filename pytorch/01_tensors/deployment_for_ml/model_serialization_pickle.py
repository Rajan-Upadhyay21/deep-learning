from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import pickle

X, y = load_iris(return_X_y=True)

model = LogisticRegression(max_iter=500)
model.fit(X, y)

with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as logistic_model.pkl")

with open("logistic_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

print("Model loaded successfully")
print("Prediction for first sample:", loaded_model.predict([X[0]]))
