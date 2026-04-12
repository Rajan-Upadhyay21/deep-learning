from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib

X, y = load_iris(return_X_y=True)

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "random_forest_model.joblib")
print("Model saved as random_forest_model.joblib")

loaded_model = joblib.load("random_forest_model.joblib")
print("Model loaded successfully")
print("Prediction for first sample:", loaded_model.predict([X[0]]))
