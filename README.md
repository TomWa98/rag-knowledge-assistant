# RAG Knowledge Assistant

A local Retrieval-Augmented Generation (RAG) application for querying and comparing technical PDF documentation using natural language.

Built with Python, ChromaDB, Sentence Transformers, Ollama, and Streamlit.

## Features

- Ask questions about technical PDF documents
- Compare specifications across multiple products
- Search individual documents or evaluate an entire document collection
- View retrieved sources with document and page references
- Run locally using an open-source language model

## Tech Stack

- **Python** — Application logic and document processing
- **ChromaDB** — Vector storage and semantic retrieval
- **Sentence Transformers** — Text embeddings (`all-MiniLM-L6-v2`)
- **Ollama** — Local LLM inference (`qwen3:1.7b`)
- **Streamlit** — Chat interface

## Getting Started

Install the Python dependencies from the project's dependency file and ensure Ollama is running with the required model.

```bash
ollama pull qwen3:1.7b
python src/ingest.py
streamlit run src/app.py
```

Place PDF documents in `data/documents/` before running ingestion.

## Retrieval Strategies

**Targeted Search:** Retrieves relevant passages from explicitly mentioned documents, or performs a semantic search across the collection.

**Exhaustive Search:** Evaluates each document individually to support collection-wide questions and comparisons.

Exhaustive Search examines every selected document but does not guarantee that every relevant passage is retrieved.

## Testing

```bash
python -m unittest discover -s tests
```

## Project Status

Active development. Current focus: improving retrieval accuracy, evidence validation, and reliable cross-document reasoning.