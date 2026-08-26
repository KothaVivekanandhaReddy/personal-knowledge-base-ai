# Changelog

## Personal Knowledge Base AI

---

## Current Milestone

### Completed: End-to-End Hybrid RAG API with Docker

The project can now:

- Load multiple PDF documents
- Extract text page by page
- Split text into overlapping chunks
- Preserve chunk metadata such as document name and page number
- Generate embeddings using Sentence Transformers
- Index document embeddings using FAISS
- Perform semantic similarity search
- Perform keyword search using a custom BM25 implementation
- Combine semantic and keyword retrieval into hybrid retrieval
- Rerank retrieved chunks using a cross-encoder reranker
- Retrieve the most relevant chunks for a query
- Send retrieved context to a local LLM
- Generate source-grounded answers
- Return source document names and page numbers
- Reject questions that are not sufficiently supported by the documents
- Avoid returning irrelevant sources when no answer is found
- Expose the RAG pipeline through a FastAPI backend
- Test the API using Swagger/OpenAPI documentation
- Containerize the application using Docker
- Run the RAG API successfully inside a Docker container
- Connect the Dockerized backend to Ollama running on the host machine

---

## Backend Components

### `pdf_loader.py`

Implemented PDF text extraction.

Responsibilities:

- Open PDF files
- Extract text page by page
- Preserve page information

---

### `chunker.py`

Implemented document chunking.

Responsibilities:

- Split extracted text into smaller chunks
- Use overlap between chunks
- Prepare text for embedding and retrieval

Current configuration:

- Chunk size: `1000`
- Overlap: `200`

---

### `ingestion.py`

Implemented the document ingestion pipeline.

Responsibilities:

- Find PDF documents
- Load PDFs
- Extract page text
- Chunk document text
- Attach metadata
- Generate unique chunk IDs

Current chunk metadata includes:

- `chunk_id`
- `document_name`
- `page_number`
- `text`

Current indexed documents:

- 5 PDF files
- Total chunks: `13,244`

---

### `embeddings.py`

Implemented embedding generation using Sentence Transformers.

Responsibilities:

- Load embedding model
- Convert document chunks into dense vectors
- Convert user queries into dense vectors
- Support semantic retrieval

---

### `vector_store.py`

Implemented FAISS-based vector retrieval.

Current design:

- Embeddings converted to `float32`
- Embeddings normalized using `faiss.normalize_L2`
- FAISS `IndexFlatIP` used for search
- Inner product on normalized vectors behaves as cosine similarity
- Chunk metadata stored separately
- Top-K retrieval supported
- FAISS index persisted to disk
- Chunk metadata persisted to JSON

Current output files:

- `data/faiss_index/index.faiss`
- `data/faiss_index/chunks.json`

---

### `bm25.py`

Implemented BM25 keyword retrieval from scratch.

Implemented components:

- Tokenization
- Term frequency calculation
- Document frequency calculation
- IDF calculation
- BM25 scoring
- Top-K keyword retrieval

The BM25 index can be saved and loaded using pickle.

Current output:

- `data/bm25_index.pkl`

---

### `hybrid_search.py`

Implemented hybrid retrieval.

Current pipeline:

Query

→ FAISS semantic search

+

→ BM25 keyword search

↓

Merge candidates

↓

Remove duplicates

↓

Cross-encoder reranking

↓

Return Top-K relevant chunks

This allows the system to combine:

- Semantic similarity
- Exact keyword matching
- Reranking for better relevance

---

### Reranker

Added a cross-encoder reranker after the initial retrieval stage.

Responsibilities:

- Receive the user query
- Compare the query with candidate chunks
- Score query-document relevance
- Reorder retrieved chunks
- Return the most relevant context to the RAG pipeline

Current retrieval architecture:

Retrieve many candidates

↓

Hybrid retrieval

↓

Rerank candidates

↓

Return best chunks

---

### `llm.py`

Implemented Python integration with Ollama.

Current configuration:

- Local Ollama API
- Model: `gemma3:1b`

When running directly on the host machine:

`http://localhost:11434/api/generate`

When running inside Docker:

`http://host.docker.internal:11434/api/generate`

The Docker networking configuration was required because `localhost` inside a Docker container refers to the container itself, not the Windows host.

---

### `rag.py`

Implemented the complete RAG pipeline.

Current pipeline:

Question

↓

Hybrid retrieval

↓

FAISS semantic candidates

+

BM25 keyword candidates

↓

Merge and remove duplicates

↓

Reranking

↓

Build context from top chunks

↓

Construct grounded prompt

↓

Send prompt to Gemma through Ollama

↓

Generate answer

↓

Check for unsupported or not-found responses

↓

Return answer and sources

The system is designed to answer only from retrieved document context.

For unsupported questions, it returns:

`I could not find the answer in the provided documents.`

When no supported answer is found:

- No hallucinated answer is returned
- Sources are returned as an empty list

---

### `main.py`

Implemented the FastAPI backend.

Current functionality:

- Initialize the RAG pipeline
- Expose the API
- Accept user questions
- Call the RAG pipeline
- Return structured JSON responses

Current endpoints:

- `GET /`
- `POST /query`

Swagger/OpenAPI documentation:

- `/docs`

The API endpoint was successfully tested and returned grounded answers with sources.

---

## Indexing Pipeline

The indexing process is now separated from the RAG application startup.

Current flow:

PDF Documents

↓

Document Ingestion

↓

Text Extraction

↓

Chunking

↓

Embeddings

↓

FAISS Index

+

BM25 Index

↓

Save Indexes to Disk

The indexes are built using:

```bash
python -m backend.build_index

Current Limitation

The system currently uses pre-built FAISS and BM25 indexes.

When new documents are added, the indexes must currently be rebuilt manually using:

python -m backend.build_index

The current system does not yet support:

Runtime document upload
Incremental indexing
Background ingestion
User authentication
Chat history
Database persistence
Redis caching
Evaluation pipelines
Observability
Cloud deployment
CI/CD

These are the next stages for evolving the project from a working end-to-end RAG system into a more complete production-style AI application.