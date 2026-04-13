# RAG Basics in Python

This folder contains Python programs focused on **RAG basics**, which are important for understanding how retrieval-augmented generation systems work in modern AI applications.

RAG stands for Retrieval-Augmented Generation. It is a workflow where an AI system first retrieves relevant external information and then uses that information to generate a response. This approach helps improve factual grounding, reduce hallucinations, and make AI responses more relevant to the user’s query.

The main purpose of this folder is to provide practical Python examples that introduce the core building blocks of RAG systems in a simple and structured way. These examples help build understanding of document chunking, embeddings, vector similarity, retrieval, context injection, reranking, metadata filtering, and simple RAG pipeline design.

This folder is designed to help learners understand how retrieval-based AI systems are structured before moving into more advanced RAG applications, vector databases, agentic systems, and production LLM pipelines.

## Why RAG is Important

RAG is important because many AI applications need access to external knowledge instead of relying only on model memory.

It is widely used in:

- enterprise knowledge assistants
- document Q&A systems
- search copilots
- customer support bots
- policy and compliance assistants
- internal company AI tools
- educational assistants
- research assistants

RAG became highly important because it can:

- improve answer grounding
- connect AI systems to fresh or domain-specific data
- reduce unsupported answers
- make AI systems more useful in real workflows

## Main Objective of This Folder

The main objective of this folder is to demonstrate how common RAG basics are implemented in Python using practical examples.

This folder helps explain:

- how documents are split into chunks
- how embeddings represent text meaning
- how similarity search retrieves useful context
- how retrievers select relevant passages
- how retrieved context is injected into prompts
- how reranking improves retrieval quality
- how metadata filters narrow results
- how a simple RAG pipeline works end to end

## Topics Covered in This Folder

This folder includes practical examples related to:

- document chunking
- text embeddings
- vector similarity search
- retrieval
- context injection
- simple RAG pipeline
- query expansion
- reranking
- metadata filtering
- RAG evaluation

These topics form a strong foundation for understanding retrieval-augmented AI workflows.

## Real-World Importance of RAG

RAG is widely used in practical AI systems because many modern assistants need reliable access to documents and knowledge sources.

For example:

- company assistants retrieve internal documentation
- customer support systems retrieve help center articles
- legal and compliance assistants retrieve policy text
- research tools retrieve relevant notes and papers
- search assistants retrieve the best matching sources before answering

These applications make RAG a highly valuable practical skill.

## What You Will Learn from This Folder

By studying and running the files in this folder, you will learn how to:

- split documents into useful retrieval chunks
- represent text with embeddings
- compare vectors for semantic similarity
- retrieve relevant passages for a query
- inject retrieved context into prompts
- improve retrieval with reranking and filtering
- structure a simple retrieval-augmented pipeline
- think about grounded AI systems more practically

This folder is especially useful because RAG is one of the most important patterns in modern AI applications.

## Folder Structure

```bash
rag_basics/
│
├── README.md
├── document_chunking_demo.py
├── text_embedding_demo.py
├── vector_similarity_search.py
├── retriever_demo.py
├── context_injection_demo.py
├── simple_rag_pipeline.py
├── query_expansion_demo.py
├── reranking_basics.py
├── metadata_filtering_demo.py
└── rag_evaluation_demo.py
