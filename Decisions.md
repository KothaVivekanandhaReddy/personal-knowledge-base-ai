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

Embedding model benchmarking may be added later.

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

Replace Chroma with FAISS.

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