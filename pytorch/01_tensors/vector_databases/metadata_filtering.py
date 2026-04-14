documents = [
    {"id": 1, "topic": "llm", "text": "Transformers are important for LLMs."},
    {"id": 2, "topic": "cv", "text": "CNNs are useful for image tasks."},
    {"id": 3, "topic": "llm", "text": "RAG improves grounded responses."},
    {"id": 4, "topic": "mlops", "text": "Monitoring helps production ML systems."}
]

selected_topic = "llm"
filtered_documents = [doc for doc in documents if doc["topic"] == selected_topic]

print("Filtered Documents:")
for doc in filtered_documents:
    print(doc)
