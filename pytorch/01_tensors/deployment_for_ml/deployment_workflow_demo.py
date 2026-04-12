def train_model():
    print("Step 1: Train model")
    return {"model": "trained_model_v1"}

def save_model(model):
    print("Step 2: Save model")
    return "trained_model_v1.pkl"

def deploy_model(model_path):
    print("Step 3: Deploy model")
    return {"status": "deployed", "path": model_path}

def serve_prediction():
    print("Step 4: Serve predictions")
    return {"prediction_status": "active"}

model = train_model()
model_path = save_model(model)
deployment_status = deploy_model(model_path)
serving_status = serve_prediction()

print("\nDeployment Workflow Result:")
print(deployment_status)
print(serving_status)
