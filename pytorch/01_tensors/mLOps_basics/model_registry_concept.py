model_registry = {
    "customer_churn_model": {
        "latest_version": "v3",
        "staging": "v2",
        "production": "v3"
    },
    "fraud_detection_model": {
        "latest_version": "v5",
        "staging": "v4",
        "production": "v5"
    }
}

print("Model Registry:")
for model_name, metadata in model_registry.items():
    print(model_name, "->", metadata)
