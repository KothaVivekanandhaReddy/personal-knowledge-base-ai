## Personal Knowledge Base AI

An end-to-end Retrieval-Augmented Generation (RAG) system for asking
questions over a personal collection of PDF documents.

The project was built from the retrieval primitives upward first, then
evaluated and deployed as a containerized application. The current V1
focuses on a reliable, measurable RAG foundation; production hardening
and framework-based comparisons are planned for V2.

## Architecture

                         User
                           |
                           v
                    +-------------+
                    |   FastAPI   |
                    |    :8000    |
                    +------+------+
                           |
                           v
                    +-------------+
                    | RAGPipeline |
                    +------+------+
                           |
                           v
                 +--------------------+
                 | FAISS VectorStore  |
                 | Dense Retrieval     |
                 +---------+----------+
                           |
                    similarity >= 0.55
                           |
                           v
                 +--------------------+
                 |  ContextBuilder    |
                 |  Top 3 chunks      |
                 |  <= 6000 chars     |
                 +---------+----------+
                           |
                           v
                 +--------------------+
                 |   PromptBuilder    |
                 +---------+----------+
                           |
                           | HTTP
                           v
                 +--------------------+
                 |      Ollama        |
                 |    Gemma 3 1B      |
                 +--------------------+
                           |
                           v
                    Answer + Sources

## What the system does

Loads PDF documents.

Extracts page-level text.

Splits documents into chunks with metadata.

Generates dense embeddings using Sentence Transformers.

Stores embeddings in a FAISS index.

Retrieves the top-K semantically similar chunks.

Applies a similarity threshold to decide whether sufficient evidence
exists.

Builds a bounded context from the highest-ranked chunks.

Generates an answer using Gemma 3 1B through Ollama.

Returns the answer together with document/page source attribution.

Abstains when the retrieved evidence is below the configured
confidence threshold.

## V1 implementation

Ingestion and indexing

PDF text extraction with pypdf

Chunking with document/page metadata

Sentence Transformer embeddings

FAISS IndexFlatIP

L2-normalized vectors for cosine-similarity retrieval

Incremental chunk indexing support

The current index contains 13,244 chunks.

## Retrieval experiments

Three retrieval configurations were implemented and evaluated:

Dense FAISS retrieval

FAISS + BM25 hybrid retrieval using Reciprocal Rank Fusion (RRF)

FAISS + BM25 + cross-encoder reranking

## Retrieval benchmark

Current evaluation uses an 8-query dataset with document-level relevance
labels.

Retrieval pipeline            Recall@5        MRR@5   Average latency

FAISS                       1.0000   1.0000    ~28.98 ms
FAISS + BM25                    1.0000       0.9375       ~100.69 ms
FAISS + BM25 + reranker         1.0000       0.9375      1348.47 ms ms

Based on this benchmark, FAISS-only was selected as the V1 serving
path because it achieved the best measured MRR and lowest measured
retrieval latency while matching the other approaches on Recall@5.

The benchmark contains only 8 queries, so these numbers are an
experimental result for the current evaluation set, not a claim of
universal retrieval superiority.

BM25, hybrid retrieval, and reranking implementations are retained for
further experimentation.

## Grounding and abstention

The RAG pipeline uses a retrieval similarity threshold of 0.55.

If the highest retrieved similarity is below the threshold, the system
does not send unsupported context to the generator. Instead it returns:

I could not find the answer in the provided documents.

For supported questions, the system returns source document names and
page numbers.

Example behavior:

Known question
    -> retrieve relevant chunks
    -> generate grounded answer
    -> return sources

Unsupported question
    -> low retrieval confidence
    -> abstain
    -> return no sources

## API

# Health check

GET /health

Example:

{
  "status": "healthy"
}

Query

POST /query
Content-Type: application/json

Request:

{
  "question": "What is machine learning engineering?"
}

Response:

{
  "answer": "...",
  "sources": [
    {
      "document_name": "Machine Learning Engineering.pdf",
      "page_number": 25
    }
  ]
}

Interactive API documentation is available through FastAPI/Swagger at:

http://localhost:8000/docs

## Running locally

Create and activate a Python virtual environment, then install:

pip install -r requirements.txt

The local development environment requires Ollama with the configured
model:

gemma3:1b

# Run the API with:

uvicorn backend.main:app --host 0.0.0.0 --port 8000

## Docker deployment

The V1 deployment uses Docker Compose with two services:

personal-knowledge-rag
    |
    +-- FastAPI
    +-- FAISS
    +-- Sentence Transformers
    +-- CPU-only PyTorch

ollama
    |
    +-- Gemma 3 1B

Build:

docker compose build

Start:

docker compose up -d

Check services:

docker ps

Health check:

curl http://localhost:8000/health

The RAG API communicates with Ollama over the Docker Compose network.


## Project structure

personal-knowledge-base-ai/
│
├── backend/
│   ├── bm25.py
│   ├── build_index.py
│   ├── chunker.py
│   ├── context_builder.py
│   ├── embeddings.py
│   ├── evaluate_retrieval.py
│   ├── hybrid_search.py
│   ├── ingestion.py
│   ├── llm.py
│   ├── main.py
│   ├── pdf_loader.py
│   ├── prompt_builder.py
│   ├── rag.py
│   ├── reranker.py
│   ├── semantic_search.py
│   ├── similarity.py
│   └── vector_store.py
│
├── evaluation/
│   ├── candidate_relevance.json
│   ├── evaluate_faiss.py
│   ├── evaluate_hybrid.py
│   ├── evaluate_reranker.py
│   ├── evaluation_dataset.json
│   ├── generate_candidates.py
│   ├── label_candidates.py
│   └── metrics.py
│
├── data/
│   ├── documents/
│   └── faiss_index/
│
├── frontend/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── ChangeLog.md
├── Decisions.md
├── Flow.md
├── experiment.txt
└── README.md


## Engineering decisions

Why FAISS for V1?

FAISS provides a simple and controllable dense retrieval baseline with
low retrieval latency. It also makes the retrieval pipeline transparent
enough to inspect, benchmark, and modify directly.

# Why keep BM25 and reranking?

They are valuable retrieval alternatives and experiments. Keeping them
in the repository allows future comparisons without making the V1
serving path unnecessarily complex.

# Why not use a RAG framework in V1?

The first implementation intentionally uses the underlying primitives
directly. This makes the retrieval, context construction, prompting, and
inference flow explicit and easier to benchmark.

A framework-based implementation will be built separately in V2 and
compared against the custom implementation using the same evaluation
methodology.


## V2  --- production engineering

V2 will evolve the V1 RAG backend into a production-oriented AI
application.

# API and reliability

Structured error handling

Configuration management

API versioning

Request IDs / correlation IDs

Timeouts and retries

Rate limiting

# Observability

Structured application logs

Request latency

Retrieval latency

LLM latency

Error metrics

Request tracing

# Evaluation

Expand the evaluation dataset

Automated retrieval regression tests

RAG answer-quality evaluation

Track benchmark results across experiments

Document platform

Document upload API

Document lifecycle management

Incremental indexing

Document deletion

Asynchronous ingestion/indexing

# Multi-user application

Authentication

Users

User/document isolation

Database-backed document metadata

Conversation history

Chat memory

# Production infrastructure

Automated tests

CI/CD

Secrets management

Caching

Cloud deployment

Scaling strategy

# Custom RAG vs framework-based RAG

A major V2 experiment is to implement the same RAG system using an
established RAG/orchestration framework and compare it against this
custom implementation.

The comparison will use the same data and evaluation set wherever
possible.

Planned dimensions:

Recall@K

MRR

Answer quality

End-to-end latency

Retrieval latency

Memory usage

Implementation complexity

Observability

Extensibility

Debuggability

Production integration effort

Control versus abstraction

The goal is not to prove that one approach is universally better. The
goal is to understand when direct implementation is preferable and
when framework abstractions provide enough value to justify their
complexity.

# Why this project?

This project is intended as an engineering study of how a RAG system
evolves from retrieval primitives into a deployable AI application.

The emphasis is on:

Build
  -> Measure
  -> Diagnose
  -> Change
  -> Benchmark
  -> Decide
  -> Deploy

rather than simply assembling a RAG pipeline from pre-built
abstractions.

Limitations

Current V1 is not a production-scale multi-tenant system.

Known limitations include:

small retrieval evaluation dataset;

document-level relevance labels;

local FAISS index;

local Ollama inference;

no authentication;

no persistent application database;

no asynchronous ingestion system;

limited observability;

no automated CI/CD pipeline;

no cloud deployment;

no horizontal scaling.

These are intentionally treated as the next engineering phase rather
than hidden from the project.
