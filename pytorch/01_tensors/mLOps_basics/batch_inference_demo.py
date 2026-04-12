import numpy as np

def mock_model_predict(batch):
    return [round(x * 1.5, 2) for x in batch]

input_batch = np.array([10, 20, 30, 40, 50])
predictions = mock_model_predict(input_batch)

print("Input Batch:")
print(input_batch)

print("\nPredictions:")
print(predictions)
