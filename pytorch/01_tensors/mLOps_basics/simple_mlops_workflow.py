def validate_data(data):
    print("Validating data...")
    return True

def train_model(data):
    print("Training model...")
    return {"model_version": "v1.0", "status": "trained"}

def register_model(model):
    print("Registering model...")
    return {"registry_status": "registered", "version": model["model_version"]}

def deploy_model(model_info):
    print("Deploying model...")
    return {"deployment_status": "success", "version": model_info["version"]}

sample_data = [1, 2, 3, 4, 5]

if validate_data(sample_data):
    model = train_model(sample_data)
    registry_info = register_model(model)
    deployment_info = deploy_model(registry_info)

    print("\nFinal Workflow Output:")
    print(deployment_info)
