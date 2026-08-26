# Personal Knowledge Base AI

A production-oriented **Retrieval-Augmented Generation (RAG)** system that allows users to query a collection of PDF documents and receive source-grounded answers.

Unlike a simple chatbot that relies only on an LLM's internal knowledge, this system retrieves relevant information from user-provided documents before generating an answer.

The project was built by implementing the core RAG pipeline step by step, starting from PDF ingestion and semantic search and progressing toward a hybrid retrieval architecture with reranking, API serving, and Docker containerization.

---

# Features

The current system supports:

- Multiple PDF documents
- PDF text extraction
- Page-level metadata tracking
- Text chunking with overlap
- Dense embeddings using Sentence Transformers
- FAISS vector similarity search
- BM25 sparse retrieval
- Hybrid retrieval
- Cross-encoder reranking
- Persistent FAISS index
- Persistent BM25 index
- Source document tracking
- Source page tracking
- Context-grounded answer generation
- Local LLM inference using Ollama
- FastAPI REST API
- Interactive API documentation
- Docker containerization

---

# System Architecture

The system separates the workflow into two major stages:

1. Offline indexing
2. Online query serving

```text
                    OFFLINE INDEXING

PDF Documents
      │
      ▼
PDF Loader
      │
      ▼
Page Extraction
      │
      ▼
Text Chunking
      │
      ▼
Metadata Generation
      │
      ├───────────────────────────────┐
      │                               │
      ▼                               ▼
Embedding Model                    BM25
      │                               │
      ▼                               ▼
FAISS Vector Index                BM25 Index
      │                               │
      ▼                               ▼
Save FAISS Index + Metadata       Save BM25 Index
```

During application startup, the persisted indexes are loaded instead of rebuilding embeddings for all documents.

```text
                     ONLINE QUERY PIPELINE

User Question
      │
      ▼
Dense Retrieval ──────────────┐
FAISS                         │
                              │
                              ▼
User Question              Hybrid Search
      │                       │
      ▼                       ▼
Sparse Retrieval ───────── Candidate Chunks
BM25                           │
                               ▼
                        Cross-Encoder
                          Reranking
                               │
                               ▼
                        Top Relevant Chunks
                               │
                               ▼
                         Context Builder
                               │
                               ▼
                         Grounded Prompt
                               │
                               ▼
                         Ollama LLM
                               │
                               ▼
                    Answer + Source Pages
```

---

# Project Structure

```text
personal-knowledge-base-ai/
│
├── backend/
│   │
│   ├── __init__.py
│   │
│   ├── main.py
│   │   FastAPI application and API endpoints
│   │
│   ├── rag.py
│   │   End-to-end RAG orchestration
│   │
│   ├── pdf_loader.py
│   │   PDF text extraction
│   │
│   ├── chunker.py
│   │   Text chunking with overlap
│   │
│   ├── ingestion.py
│   │   Document ingestion and metadata creation
│   │
│   ├── embeddings.py
│   │   Sentence Transformer embedding generation
│   │
│   ├── similarity.py
│   │   Cosine similarity implementation from scratch
│   │
│   ├── semantic_search.py
│   │   Initial semantic search implementation
│   │
│   ├── vector_store.py
│   │   FAISS vector indexing and retrieval
│   │
│   ├── bm25.py
│   │   BM25 sparse retrieval implementation
│   │
│   ├── hybrid_search.py
│   │   Dense + sparse retrieval
│   │
│   ├── reranker.py
│   │   Cross-encoder reranking
│   │
│   ├── llm.py
│   │   Ollama HTTP API integration
│   │
│   └── build_index.py
│       Offline index building pipeline
│
├── data/
│   │
│   ├── documents/
│   │   PDF documents used as the knowledge base
│   │
│   ├── faiss_index/
│   │   ├── index.faiss
│   │   └── chunks.json
│   │
│   └── bm25_index.pkl
│
├── requirements.txt
│
├── Dockerfile
│
├── .dockerignore
│
├── CHANGELOG.md
│
├── DECISIONS.md
│
├── FLOW.md
│
└── README.md
```

---

# Documents Currently Indexed

The current knowledge base contains multiple PDF documents.

During the latest successful index build:

```text
Total PDF files: 5

Total chunks: 13,244
```

The indexed documents include material related to:

- Machine Learning
- Applied Machine Learning and AI Engineering
- Deep Learning
- Machine Learning Engineering
- Physics

The system is designed so that additional PDF files can be added to:

```text
data/documents/
```

After adding new documents, the indexes must be rebuilt.

---

# Core Components

## 1. PDF Loader

File:

```text
backend/pdf_loader.py
```

Responsible for:

- Opening PDF documents
- Extracting text
- Processing documents page by page
- Preserving page numbers

The page number is stored as metadata and later returned with generated answers.

---

# 2. Chunking

File:

```text
backend/chunker.py
```

Documents are split into smaller text chunks before indexing.

Current chunking approach:

```text
Chunk Size: 1000 characters
Overlap: 200 characters
```

Conceptually:

```text
Document Text

[--------- Chunk 1 ---------]
                  [--------- Chunk 2 ---------]
                                    [--------- Chunk 3 ---------]
```

Overlap helps preserve context between neighboring chunks.

---

# 3. Document Ingestion

File:

```text
backend/ingestion.py
```

The ingestion pipeline processes all PDFs in the documents directory.

Each generated chunk contains metadata such as:

```python
{
    "chunk_id": ...,
    "document_name": ...,
    "page_number": ...,
    "text": ...
}
```

This metadata is preserved throughout retrieval.

---

# 4. Embeddings

File:

```text
backend/embeddings.py
```

The system uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model converts text into dense vector representations.

Embedding dimension:

```text
384
```

Embeddings are generated for:

- Document chunks during indexing
- User queries during retrieval

Conceptually:

```text
"machine learning"

        ↓

Embedding Model

        ↓

[0.21, -0.34, 0.87, ...]
```

---

# 5. FAISS Vector Search

File:

```text
backend/vector_store.py
```

Dense vector retrieval is implemented using FAISS.

The workflow is:

```text
Document Chunks
      ↓
Generate Embeddings
      ↓
Convert to float32
      ↓
L2 Normalize
      ↓
FAISS IndexFlatIP
      ↓
Save Index
```

The system uses:

```python
faiss.IndexFlatIP
```

with normalized embeddings.

After normalization:

```text
cosine_similarity(a, b) = dot_product(a, b)
```

Therefore, inner product search behaves like cosine similarity ranking.

The FAISS index and chunk metadata are persisted to disk.

```text
data/faiss_index/index.faiss
data/faiss_index/chunks.json
```

This avoids regenerating embeddings every time the API starts.

---

# 6. BM25 Sparse Retrieval

File:

```text
backend/bm25.py
```

BM25 provides lexical retrieval.

It is useful when the user's query contains:

- Exact terminology
- Technical keywords
- Rare words
- Specific names
- Terms that dense retrieval may not prioritize strongly

Example:

```text
Query:

"What is gradient descent?"
```

BM25 searches for chunks containing relevant lexical terms such as:

```text
gradient
descent
```

The BM25 index is persisted to:

```text
data/bm25_index.pkl
```

---

# 7. Hybrid Retrieval

File:

```text
backend/hybrid_search.py
```

The system combines two retrieval strategies.

```text
                    User Query
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
       Dense Search            Sparse Search
          FAISS                    BM25
            │                       │
            └───────────┬───────────┘
                        │
                        ▼
                  Candidate Chunks
```

Dense retrieval helps find semantically similar content.

BM25 helps find lexically relevant content.

Combining both strategies improves retrieval robustness compared with relying on only one retrieval method.

---

# 8. Cross-Encoder Reranking

File:

```text
backend/reranker.py
```

Initial retrieval generates a set of candidate chunks.

A cross-encoder then evaluates the relationship between:

```text
Question + Candidate Chunk
```

Each candidate is scored again based on direct relevance to the query.

Pipeline:

```text
Initial Retrieval
      │
      ▼
Candidate Chunks
      │
      ▼
Cross-Encoder Reranker
      │
      ▼
Reordered Results
      │
      ▼
Top Relevant Context
```

This is more computationally expensive than vector similarity search, so reranking is applied after candidate retrieval rather than across all 13,244 chunks.

---

# 9. RAG Pipeline

File:

```text
backend/rag.py
```

The RAG pipeline coordinates retrieval and generation.

The current query flow is:

```text
Question
   ↓
Hybrid Retrieval
   ↓
Candidate Chunks
   ↓
Cross-Encoder Reranking
   ↓
Top Relevant Results
   ↓
Context Construction
   ↓
Prompt Construction
   ↓
Ollama LLM
   ↓
Generated Answer
   ↓
Sources
```

The context sent to the model contains source information.

Example:

```text
SOURCE 1

Document: Applied-Machine-Learning.pdf

Page: 42

<retrieved text>
```

This allows the application to return both:

```text
Answer
```

and:

```text
Document + Page Number
```

---

# 10. Grounded Generation

The LLM is explicitly instructed to answer using only the retrieved context.

The prompt contains rules similar to:

```text
Answer the question using ONLY the provided context.

Do not use outside knowledge.

If the answer is not contained in the context, say:

"I could not find the answer in the provided documents."
```

This reduces unsupported answers.

The system also checks for no-answer responses and returns:

```json
{
    "answer": "I could not find the answer in the provided documents.",
    "sources": []
}
```

when appropriate.

---

# 11. Local LLM

File:

```text
backend/llm.py
```

The system currently uses Ollama for local inference.

Current model:

```text
gemma3:1b
```

For local development:

```text
http://localhost:11434/api/generate
```

When running inside Docker:

```text
http://host.docker.internal:11434/api/generate
```

This distinction is necessary because:

```text
localhost inside a container
```

refers to the container itself, not the host machine.

The architecture therefore looks like:

```text
Docker Container
      │
      │ HTTP Request
      ▼
host.docker.internal
      │
      ▼
Host Machine
      │
      ▼
Ollama Server
      │
      ▼
gemma3:1b
```

---

# 12. FastAPI Backend

File:

```text
backend/main.py
```

The application exposes the RAG system through a REST API.

The server can be started locally using:

```bash
uvicorn backend.main:app --reload
```

The API becomes available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

The main query endpoint is:

```text
POST /query
```

The endpoint receives a question and returns a generated answer with sources.

Example response structure:

```json
{
    "answer": "Generated answer from the retrieved documents.",
    "sources": [
        {
            "document_name": "example.pdf",
            "page_number": 42
        }
    ]
}
```

The API was tested successfully both locally and through Docker.

---

# Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd personal-knowledge-base-ai
```

---

# 2. Create a Virtual Environment

Windows:

```bash
python -m venv env
```

Activate:

```bash
env\Scripts\activate
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Install and Run Ollama

Install Ollama and download the model:

```bash
ollama pull gemma3:1b
```

Verify available models:

```bash
ollama list
```

The Ollama service should be available on:

```text
http://localhost:11434
```

---

# Building the Index

Place PDF documents inside:

```text
data/documents/
```

Then run:

```bash
python -m backend.build_index
```

The indexing pipeline performs:

```text
PDF Documents
      ↓
Text Extraction
      ↓
Chunking
      ↓
Embedding Generation
      ↓
FAISS Index Creation
      ↓
FAISS Index Persistence
      ↓
BM25 Index Creation
      ↓
BM25 Index Persistence
```

A successful run currently produces:

```text
13,244 indexed chunks
```

The generated indexes are stored in:

```text
data/faiss_index/
```

and:

```text
data/bm25_index.pkl
```

---

# Running Locally

After the indexes have been created:

```bash
uvicorn backend.main:app --reload
```

Expected startup flow:

```text
Initializing RAG pipeline...

Loading FAISS vector store...
Loading FAISS index...
Loading chunk metadata...

Loading BM25 index...

Loading reranker...
Reranker ready.

Initializing LLM...

RAG pipeline ready.
Application ready.
```

Open:

```text
http://127.0.0.1:8000/docs
```

Use the `/query` endpoint to test the system.

---

# Docker

The project is containerized using Docker.

## Build the Image

From the project root:

```bash
docker build -t personal-knowledge-rag .
```

Docker performs the following operations:

```text
Dockerfile
    ↓
Python Base Image
    ↓
Install Dependencies
    ↓
Copy Application
    ↓
Create Container Image
```

The final image is:

```text
personal-knowledge-rag
```

---

# Run the Container

```bash
docker run --rm -p 8000:8000 personal-knowledge-rag
```

The API will be available at:

```text
http://localhost:8000
```

Interactive documentation:

```text
http://localhost:8000/docs
```

---

# Important Docker Networking Decision

The RAG application runs inside a Docker container, while Ollama currently runs on the host machine.

Therefore, this will not work from inside the container:

```text
http://localhost:11434
```

Inside Docker, `localhost` refers to the container itself.

The application instead uses:

```text
http://host.docker.internal:11434/api/generate
```

Architecture:

```text
Browser
   │
   ▼
localhost:8000
   │
   ▼
Docker Container
   │
   │
   ├── FastAPI
   ├── FAISS
   ├── BM25
   ├── Reranker
   └── RAG Pipeline
          │
          ▼
host.docker.internal:11434
          │
          ▼
Host Machine Ollama
          │
          ▼
gemma3:1b
```

This configuration was required because the first containerized version attempted to access:

```text
localhost:11434
```

and failed with:

```text
ConnectionRefusedError
```

The endpoint was updated to:

```text
host.docker.internal
```

so the container can communicate with Ollama running on the host machine.

---

# API Flow

A request to:

```text
POST /query
```

follows this pipeline:

```text
User Question
      │
      ▼
FastAPI Endpoint
      │
      ▼
RAGPipeline
      │
      ▼
HybridSearch
      │
      ├───────────────┐
      ▼               ▼
FAISS              BM25
Dense              Sparse
Search             Search
      │               │
      └───────┬───────┘
              ▼
        Candidate Chunks
              │
              ▼
          Reranker
              │
              ▼
       Top Relevant Chunks
              │
              ▼
        Context Builder
              │
              ▼
         Prompt Builder
              │
              ▼
           Ollama
              │
              ▼
       Generated Answer
              │
              ▼
    Answer + Source Metadata
```

---

# Example Query

Request:

```json
{
    "question": "What is machine learning?"
}
```

The application:

1. Embeds the query.
2. Searches FAISS for semantically similar chunks.
3. Searches BM25 for lexically relevant chunks.
4. Combines candidate results.
5. Reranks the candidates using a cross-encoder.
6. Selects the most relevant chunks.
7. Builds a source-aware context.
8. Sends the context and question to the LLM.
9. Returns the generated answer.
10. Returns the associated document and page sources.

---

# Why Hybrid Retrieval?

Dense retrieval and sparse retrieval have different strengths.

## Dense Retrieval

Useful for:

```text
semantic similarity
conceptual similarity
different wording
paraphrases
```

Example:

```text
Query:
"How does a model learn from examples?"
```

May retrieve:

```text
"Machine learning algorithms identify patterns from training data."
```

even if the exact words are different.

---

## Sparse Retrieval

Useful for:

```text
exact keywords
technical terminology
rare terms
specific names
```

Example:

```text
Query:
"backpropagation"
```

BM25 can strongly prioritize chunks containing that specific term.

---

## Hybrid Retrieval

Combines both:

```text
Semantic Retrieval
        +
Lexical Retrieval
        =
More Robust Candidate Retrieval
```

---

# Why Reranking?

Initial retrieval is optimized for speed.

However, the top retrieved chunks are not always ordered perfectly by actual relevance.

The cross-encoder receives:

```text
Question + Chunk
```

and directly estimates their relevance.

Therefore:

```text
Fast Retrieval
      ↓
Candidate Set
      ↓
More Expensive Relevance Evaluation
      ↓
Better Final Ranking
```

This follows a common multi-stage retrieval architecture.

---

# Engineering Decisions

Important architectural decisions are documented separately in:

```text
DECISIONS.md
```

The document includes decisions regarding:

- Embedding model selection
- Chroma to FAISS migration
- Cosine similarity implementation
- Persistent indexing
- BM25 implementation
- Hybrid retrieval
- Cross-encoder reranking
- Ollama
- FastAPI
- Docker networking

---

# Development History

The project evolution is documented in:

```text
CHANGELOG.md
```

The architecture and request/data flow are documented in:

```text
FLOW.md
```

---

# Current Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| PDF Processing | Python PDF processing library |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Dense Retrieval | FAISS |
| Sparse Retrieval | Custom BM25 |
| Retrieval | Hybrid Search |
| Reranking | Cross-Encoder |
| LLM Runtime | Ollama |
| Current LLM | gemma3:1b |
| Backend API | FastAPI |
| ASGI Server | Uvicorn |
| Containerization | Docker |
| OS Integration | WSL2 / Docker Desktop |

---

# Current Project Status

## Completed

### Document Processing

- [x] Multi-PDF ingestion
- [x] Page-level text extraction
- [x] Text chunking
- [x] Chunk overlap
- [x] Metadata preservation

### Dense Retrieval

- [x] Embedding generation
- [x] Cosine similarity understanding
- [x] Semantic search baseline
- [x] FAISS vector indexing
- [x] Top-K vector retrieval

### Sparse Retrieval

- [x] BM25 implementation
- [x] BM25 persistence
- [x] BM25 loading

### Hybrid Retrieval

- [x] Dense retrieval
- [x] Sparse retrieval
- [x] Candidate combination

### Reranking

- [x] Cross-encoder reranking
- [x] Multi-stage retrieval pipeline

### RAG

- [x] Context construction
- [x] Grounded prompting
- [x] Local LLM generation
- [x] No-answer handling
- [x] Source tracking
- [x] Page tracking

### Backend

- [x] FastAPI application
- [x] POST query endpoint
- [x] Interactive Swagger documentation
- [x] Local API testing

### Infrastructure

- [x] Persistent FAISS index
- [x] Persistent BM25 index
- [x] Dockerfile
- [x] Docker image build
- [x] Docker container execution
- [x] Host-to-container Ollama connectivity
- [x] Containerized API testing

---

# Current Limitations

The current version is a strong end-to-end RAG backend, but it is not yet a complete production system.

Current limitations include:

- No authentication
- No user accounts
- No document upload API
- No database
- No chat history
- No conversation memory
- No asynchronous background indexing
- No retrieval evaluation benchmark
- No automated test suite
- No CI/CD pipeline
- No observability or tracing
- No caching
- No rate limiting
- Ollama runs outside the application container
- No multi-container orchestration
- No cloud deployment yet

These are intentional next-stage engineering areas rather than missing fundamentals.

---

# Planned Next Steps

The next stages of the project can evolve toward a more production-oriented AI system.

Potential improvements include:

## Retrieval Evaluation

Build a benchmark dataset containing:

```text
Question
Expected Source
Expected Answer
Relevant Chunk
```

Evaluate:

- Recall@K
- Precision@K
- MRR
- Reranking impact
- Dense vs BM25 vs Hybrid retrieval

---

## API Improvements

Add:

- Request validation
- Better error handling
- Health endpoint
- Configuration management
- Environment variables
- API versioning

---

## Document Management

Add:

- PDF upload endpoint
- Background ingestion
- Incremental indexing
- Document deletion
- Index updates without rebuilding everything

---

## Application Layer

Add:

- Authentication
- Users
- Database
- Conversation history
- Chat sessions

---

## Production Infrastructure

Add:

- Docker Compose
- Ollama as a service or remote inference
- Redis caching
- PostgreSQL
- CI/CD
- Automated testing
- Logging
- Monitoring
- Tracing
- Cloud deployment

---

# Learning Approach

This project was intentionally not built by starting with a high-level RAG framework.

The architecture was developed layer by layer:

```text
PDF Loading
    ↓
Chunking
    ↓
Embeddings
    ↓
Cosine Similarity
    ↓
Semantic Search
    ↓
FAISS
    ↓
Persistent Indexing
    ↓
BM25
    ↓
Hybrid Retrieval
    ↓
Reranking
    ↓
RAG Pipeline
    ↓
FastAPI
    ↓
Docker
```

The objective was to understand the underlying system before introducing higher-level abstractions.

This makes it easier to reason about:

- Retrieval failures
- Chunking problems
- Embedding quality
- Similarity scores
- Indexing
- Reranking
- LLM grounding
- API architecture
- Container networking

---

# Current Milestone

## End-to-End Containerized RAG Backend

The current milestone has been successfully completed.

The system can:

```text
Take a user question
        ↓
Search 13,244 indexed document chunks
        ↓
Combine dense and sparse retrieval
        ↓
Rerank candidate chunks
        ↓
Construct grounded context
        ↓
Generate an answer using a local LLM
        ↓
Return document and page sources
        ↓
Serve the system through FastAPI
        ↓
Run the backend inside Docker
```
