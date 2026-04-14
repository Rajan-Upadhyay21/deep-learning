import numpy as np
import pandas as pd

store = pd.DataFrame({
    "document_id": [1, 2, 3],
    "text": [
        "PyTorch supports neural network development.",
        "RAG combines retrieval and generation.",
        "Vector databases store embeddings for search."
    ],
    "embedding": [
        np.array([0.91, 0.10, 0.20]),
        np.array([0.12, 0.35, 0.97]),
        np.array([0.20, 0.25, 0.90])
    ]
})

print("Document Embedding Store:")
print(store)
