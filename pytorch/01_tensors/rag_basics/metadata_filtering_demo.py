documents = [
    {"title": "PyTorch Guide", "topic": "deep_learning", "text": "PyTorch supports neural network development."},
    {"title": "RAG Notes", "topic": "llm", "text": "RAG combines retrieval and generation."},
    {"title": "CNN Intro", "topic": "computer_vision", "text": "CNNs are used for image tasks."},
    {"title": "Transformer Basics", "topic": "llm", "text": "Transformers are core to modern language models."}
]

selected_topic = "llm"

filtered_docs = [doc for doc in documents if doc["topic"] == selected_topic]

print("Filtered Documents:")
for doc in filtered_docs:
    print(doc)
