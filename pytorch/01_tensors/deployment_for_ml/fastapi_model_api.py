from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import numpy as np

app = FastAPI()

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

class PredictionInput(BaseModel):
    value: float

@app.post("/predict")
def predict(input_data: PredictionInput):
    prediction = model.predict(np.array([[input_data.value]]))[0]
    return {"input": input_data.value, "prediction": float(prediction)}
