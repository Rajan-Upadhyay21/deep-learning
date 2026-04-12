
Code files:

## `flask_model_api.py`

```python
from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    value = data.get("value", 0)
    prediction = model.predict(np.array([[value]]))[0]
    return jsonify({"input": value, "prediction": float(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
