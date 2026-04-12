def load_data():
    print("Step 1: Loading data")
    return [1, 2, 3, 4, 5]

def preprocess_data(data):
    print("Step 2: Preprocessing data")
    return [x * 2 for x in data]

def train_model(data):
    print("Step 3: Training model")
    model = {"trained_on": data, "status": "trained"}
    return model

def evaluate_model(model):
    print("Step 4: Evaluating model")
    return {"accuracy": 0.91, "model_status": model["status"]}

data = load_data()
processed_data = preprocess_data(data)
model = train_model(processed_data)
results = evaluate_model(model)

print("\nPipeline Result:")
print(results)
