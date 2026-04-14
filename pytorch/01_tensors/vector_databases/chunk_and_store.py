import numpy as np

text = """
RAG systems retrieve relevant chunks.
Chunking helps large documents become searchable.
Embeddings represent chunk meaning for similarity search.
""".strip()

lines = text.split("\n")
chunks = [line.strip() for line in lines if line.strip()]

store = []
for idx, chunk in enumerate(chunks, start=1):
    store.append({
        "chunk_id": f"chunk_{idx}",
        "text": chunk,
        "embedding": np.random.rand(4)
    })

print("Chunk Store:")
for item in store:
    print(item)
