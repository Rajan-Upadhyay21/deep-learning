query = "How do RAG systems improve AI answers?"

retrieved_context = """
RAG systems first retrieve relevant documents.
The retrieved context is then added to the prompt.
This helps make responses more grounded and relevant.
""".strip()

prompt = f"""
You are a helpful AI assistant.

Use the following context to answer the question.

Context:
{retrieved_context}

Question:
{query}

Answer:
""".strip()

print("Final Prompt:")
print(prompt)
