import numpy as np

vocab = ["AI", "is", "very", "powerful", "useful"]
probabilities = np.array([0.10, 0.15, 0.20, 0.40, 0.15])

selected_token = vocab[np.argmax(probabilities)]

print("Vocabulary:", vocab)
print("Probabilities:", probabilities)
print("\nGreedy Decoding Selected Token:")
print(selected_token)
