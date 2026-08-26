# Personal Knowledge Base AI

A production-oriented Retrieval-Augmented Generation (RAG) system that allows users to query a collection of PDF documents and receive source-grounded answers.

The system supports:

- Multiple PDF documents
- Persistent vector indexing
- Semantic retrieval using FAISS
- Sparse retrieval using BM25
- Hybrid retrieval
- Cross-encoder reranking
- Local LLM inference using Ollama
- Source and page tracking
- FastAPI backend
- Docker containerization

---

# Architecture

The system separates **offline indexing** from the **online query pipeline**.

## Offline Indexing

```text
PDF Documents
      ↓
PDF Loader
      ↓
Page Extraction
      ↓
Chunking
      ↓
Metadata Generation
      ↓
Embedding Model
      ↓
FAISS Vector Index
      ↓
Save FAISS Index + Metadata

Chunk Text
      ↓
BM25 Index
      ↓
Save BM25 Index
