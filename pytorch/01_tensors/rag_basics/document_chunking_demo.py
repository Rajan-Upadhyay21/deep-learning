
Code files:

## `document_chunking_demo.py`

```python
text = """
PyTorch is a popular deep learning framework.
Transformers are widely used in modern NLP systems.
RAG combines retrieval and generation.
Embeddings help semantic search.
Chunking is important for effective retrieval.
""".strip()

lines = text.split("\n")
chunks = []
chunk_size = 2

for i in range(0, len(lines), chunk_size):
    chunk = " ".join(lines[i:i + chunk_size]).strip()
    chunks.append(chunk)

print("Original Text:")
print(text)

print("\nChunks:")
for idx, chunk in enumerate(chunks, start=1):
    print(f"Chunk {idx}: {chunk}")
