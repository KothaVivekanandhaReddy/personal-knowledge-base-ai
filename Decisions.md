# Engineering Decisions

## Project: Personal Knowledge Base AI

This document records important architecture and implementation decisions.

---

# 1. Embedding Model

## Decision

Use:

`sentence-transformers/all-MiniLM-L6-v2`

## Reason

The model:

- Runs locally
- Has relatively low resource requirements
- Produces 384-dimensional embeddings
- Is sufficient for the initial RAG baseline
- Works on the current development machine

## Trade-off

It may not provide the same retrieval quality as larger or domain-specific embedding models.



---

# 2. Vector Database / Vector Search

## Initial Decision

Use Chroma.

## Problem

Chroma caused native crashes during document insertion.

Observed Windows process exit code:

`-1073741819`

The crash occurred even during a minimal add-document test.

## Final Decision

Replaced Chroma with FAISS.

## Reason

FAISS:

- Installed successfully
- Worked correctly with Python 3.12
- Successfully indexed all 685 chunks
- Provides direct control over indexing and similarity search
- Is lightweight for the current local project stage

---

# 3. Similarity Metric

## Initial Approach

FAISS `IndexFlatL2`.

## Final Decision

Use normalized embeddings with:

`faiss.IndexFlatIP`

## Reason

After L2 normalization:

cosine_similarity(a, b) = dot_product(a, b)

Therefore:

- Normalize document embeddings
- Normalize query embeddings
- Use inner product search

This produces cosine-similarity-style ranking.

---

# 4. Retrieval Strategy

## Decision

Use:

- Top-K retrieval
- Default `top_k = 3`
- Minimum similarity threshold = `0.5`

## Reason

Testing showed:

Relevant document questions:

~0.62 to ~0.71

Clearly unrelated question:

~0.22 to ~0.26

The threshold prevents weak retrieval results from being passed to the LLM.

## Important Note

`0.5` is currently an initial heuristic.

It is not considered a universal or final threshold.

Future evaluation will determine an appropriate threshold using a benchmark dataset.

---

# 5. LLM

## Decision

Use local Ollama inference.

Current model:

`gemma3:1b`

## Reason

The development machine has limited available memory during normal usage.

A small local model allows:

- Local inference
- No API cost
- Direct control over the full RAG pipeline
- Learning of local AI infrastructure

## Future Possibilities

Benchmark:

- Stronger local models
- Cloud API models
- Latency vs quality trade-offs

---

# 6. LLM Integration

## Decision

Use Ollama HTTP API directly.

Endpoint:

`http://localhost:11434/api/generate`

## Reason

Direct HTTP integration provides a clear understanding of the interface between:

Python application
→ Local inference server
→ LLM

Framework abstractions such as LangChain will not be introduced until the underlying pipeline is understood.

---

# 7. Grounded Generation

## Decision

The LLM receives retrieved document context and is instructed to answer only from that context.

Prompt behavior:

- Use only provided context
- If answer is not in context, report that the answer was not found

## Additional Protection

Retrieval thresholding occurs before LLM generation.

If no chunk exceeds the threshold:

- Do not send irrelevant context to the LLM
- Return a no-answer response

---

# 8. Source Tracking

## Decision

Preserve page numbers during ingestion.

Each retrieved result contains:

- Text
- Page number
- Similarity score

## Reason

This enables:

- Source-grounded answers
- Basic traceability
- Future citation improvements

---

# 9. Current Architecture

The project is intentionally implemented in layers:

PDF Loading
→ Chunking
→ Embeddings
→ Vector Index
→ Retrieval
→ Prompt Construction
→ LLM Generation

Each layer was tested independently before integration.

---

# 10. Next Architecture Decision

The current FAISS index exists only in memory.

Next decision:

Persist:

- FAISS index
- Chunk metadata

This will separate ingestion-time computation from query-time execution.
Vector Index Persistence
Initial Architecture

The FAISS index existed only in memory.

The application previously performed:

PDF
↓
Chunking
↓
Embedding generation
↓
FAISS index creation
↓
Application startup

This meant embeddings and indexes were rebuilt every time the application restarted.

Problem

This was inefficient because embedding thousands of chunks is an ingestion-time operation, not a query-time operation.

Final Decision

Persist:

FAISS index
Chunk metadata

Current storage:

data/faiss_index/index.faiss
data/faiss_index/chunks.json

## Sparse Retrieval
Decision

Implement BM25 retrieval from scratch.

Reason

Dense vector search and lexical search have different strengths.

Dense retrieval is useful for:

Semantic similarity
Conceptually similar wording
Queries that do not exactly match document vocabulary

BM25 is useful for:

Exact terms
Technical terminology
Keywords
Specific names
Implementation

A custom BM25 implementation calculates:

Tokenization
Term frequency
Document frequency
Inverse document frequency
BM25 scores

The BM25 index is persisted using Python serialization.

Current storage:

data/bm25_index.pkl

## Hybrid Retrieval
Decision

Combine:

Dense Retrieval
+
BM25 Retrieval
Reason

Neither retrieval method is sufficient for all query types.

Dense retrieval may miss exact technical terms.

BM25 may miss semantically related content when the query uses different wording.

Hybrid retrieval provides candidate documents from both systems.

## Reranking
Decision

Add a reranking stage after hybrid retrieval.

Reason

Initial retrieval produces candidate chunks, but the top-ranked result from dense or BM25 search is not always the most relevant answer context.

Reranking evaluates:

Query + Candidate Chunk

together.

This allows the system to perform a more precise relevance judgment.

Retrieval Strategy
Initial Strategy

Use dense vector search with:

top_k = 3

and a similarity threshold.

Current Strategy

Use hybrid retrieval followed by reranking.

The retriever:

Performs dense search
Performs BM25 search
Merges candidate results
Removes duplicates
Sends candidates to the reranker
Returns the highest-ranked chunks

Backend API
Decision

Expose the RAG pipeline using FastAPI.

Current Endpoints
Root
GET /

Used to verify that the API is running.

Query Endpoint
POST /query

Accepts a user question and sends it through the RAG pipeline.
## Reason
FastAPI provides:

API validation
Automatic OpenAPI documentation
Interactive Swagger UI
Clear backend interfaces
A foundation for future frontend integration

## Docker Containerization
Decision

Containerize the FastAPI RAG backend.

The application is built into a Docker image using:

python:3.12-slim

The image:

Sets the application working directory
Copies dependency requirements
Installs Python dependencies
Copies the project source code
Starts the FastAPI application using Uvicorn
Docker Image

Current image:

personal-knowledge-rag
Running the Container

The application exposes:

8000

and maps it to the host:

localhost:8000

The API documentation is available through:

http://localhost:8000/docs

# 2026-09-02 — V1 Architecture and Deployment Decisions

## Decision: FAISS-only retrieval for V1 serving

We evaluated three retrieval configurations:

| Pipeline | Recall@5 | MRR@5 | Avg Latency |
|---|---:|---:|---:|
| FAISS | 1.0000 | 1.0000 | ~30.29 ms |
| FAISS + BM25 | 1.0000 | 0.9375 | ~132.70 ms |
| FAISS + BM25 + Reranker | 1.0000 | 0.9375 | ~2430.32 ms |

FAISS-only was selected for V1 because it achieved the best measured MRR and lowest measured latency while matching the other approaches on Recall@5.

The current evaluation dataset contains only 8 queries, so this result is treated as an experimental benchmark rather than a universal claim that dense retrieval is always superior.

BM25, hybrid retrieval, and reranking are retained for future experimentation.

---

## Decision: Retrieval abstention threshold

A similarity threshold of `0.55` is currently used.

Observed evaluation during calibration showed a clear separation between relevant and unrelated queries on the current dataset.

The threshold prevents the generator from answering when the retrieved evidence is insufficient.

This threshold is an initial calibrated value and should be re-evaluated as the evaluation dataset grows.

---

## Decision: CPU-only PyTorch for the RAG API container

The RAG API uses Sentence Transformers for embedding inference.

The current deployment does not require GPU inference inside the RAG API container.

Therefore the Docker image explicitly installs CPU-only PyTorch.

LLM inference is delegated to the separate Ollama service.

This avoids unnecessary CUDA, cuDNN, NCCL, and related NVIDIA dependencies in the API image.

---

## Decision: Separate Ollama service

Ollama runs as a separate container from the FastAPI RAG API.

The RAG API communicates with Ollama through HTTP.

This separates:

- application/API responsibilities
- embedding/retrieval responsibilities
- LLM runtime responsibilities

and allows the LLM runtime to be changed independently from the RAG API.

---

## Decision: V1 scope is frozen

V1 is considered complete after successful end-to-end Docker validation.

Further improvements will be implemented as V2 production engineering work rather than continuously modifying the V1 architecture.