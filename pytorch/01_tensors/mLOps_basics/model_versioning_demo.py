
Code files:

## `model_versioning_demo.py`

```python
models = [
    {"version": "v1.0", "algorithm": "RandomForest", "accuracy": 0.89},
    {"version": "v1.1", "algorithm": "RandomForest", "accuracy": 0.91},
    {"version": "v2.0", "algorithm": "XGBoost", "accuracy": 0.93}
]

print("Registered Model Versions:")
for model in models:
    print(model)
