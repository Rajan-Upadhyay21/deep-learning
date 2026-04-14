import numpy as np

vector_index = {
    "chunk_1": np.array([0.91, 0.10, 0.20]),
    "chunk_2": np.array([0.12, 0.35, 0.97]),
    "chunk_3": np.array([0.20, 0.25, 0.90])
}

print("Simple Vector Index:")
for key, value in vector_index.items():
    print(f"{key}: {value}")
