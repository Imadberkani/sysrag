<h1 align="center">📰 RAG Pipeline: Retrieval-Augmented QA on BBC News</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Build%20%26%20Test-passing-2ea44f?style=flat-square" alt="Build and Test passing" />
  <img src="https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue?style=flat-square" alt="Python versions" />
  <img src="https://img.shields.io/badge/license-Apache%20Software%20License%202.0-lightgrey?style=flat-square" alt="License Apache Software License 2.0" />
</p>


<p align="center">
  <a href="./src/assets/rag_pipeline.png">
    <img
      src="./src/assets/rag_pipeline.png"
      alt="RAG Architecture Diagram"
    />
  </a>
</p>

## 🔍 Overview

Large Language Models can generate impressive responses, but they remain constrained by static training knowledge and limited context windows. When too much information is packed into a prompt, relevance drops, costs increase, and important details can be missed.

This repository showcases a **RAG pipeline built on BBC News data**, designed to retrieve, rerank, and inject only the most relevant context into the generation step. By combining **Weaviate vector search**, context building, and LLM-based answer generation, the system produces more grounded, scalable, and context-efficient responses.

<p align="center">
  <img src="./src/assets/Animation.gif" alt="RAG demo animation" width="100%" />
</p>

## 📦 Installation

This project uses **Poetry** for dependency management and environment isolation.

```bash
poetry --version
poetry init
poetry config virtualenvs.in-project true
poetry install --no-root
poetry env use python
poetry add --group dev jupyter ipykernel
poetry run python -m ipykernel install --user --name sysrag --display-name "Python (sysrag)"
```
## 📝 Instructions

```bash
poetry run python flask_app.py
docker compose up -d
```

## 🛣️ Roadmap

- [x] Build an end-to-end RAG pipeline on BBC News data
- [x] Integrate Weaviate-based vector retrieval
- [x] Add reranking for better context selection
- [x] Expose RAG pipeline endpoints through Flask
- [x] Containerize the runtime with Docker Compose
- [ ] Expose the RAG system through a user interface