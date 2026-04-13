import numpy as np

documents = [
    "PyTorch is a deep learning framework",
    "Transformers are useful for language models",
    "RAG systems use retrieval and generation"
]

vocab = sorted(set(" ".join(documents).lower().split()))
word_to_index = {word: idx for idx, word in enumerate(vocab)}

embeddings = []

for doc in documents:
    vector = np.zeros(len(vocab))
    for word in doc.lower().split():
        vector[word_to_index[word]] += 1
    embeddings.append(vector)

embeddings = np.array(embeddings)

print("Vocabulary:")
print(vocab)

print("\nEmbedding Matrix Shape:")
print(embeddings.shape)

print("\nEmbeddings:")
print(embeddings)
