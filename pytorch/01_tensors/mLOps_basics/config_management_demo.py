config = {
    "model_name": "customer_churn_model",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10,
    "environment": "staging"
}

print("Configuration:")
for key, value in config.items():
    print(f"{key}: {value}")
