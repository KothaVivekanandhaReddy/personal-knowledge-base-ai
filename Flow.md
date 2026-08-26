# System Flow

## Personal Knowledge Base AI

---

# Current End-to-End RAG Flow

```text
                    ┌─────────────────┐
                    │   PDF Document  │
                    └────────┬────────┘
                             │
                             ▼
                       PDF Loader
                             │
                             ▼
                      Extracted Pages
                             │
                             ▼
                          Chunker
                             │
                             ▼
                    Document Chunks
                             │
                             │
                             ▼
                      Embedding Model
                  all-MiniLM-L6-v2
                             │
                             ▼
                     384-d Embeddings
                             │
                             ▼
                    L2 Normalization
                             │
                             ▼
                  FAISS IndexFlatIP
                             │
                             ▼
                      Vector Index

## Query Flow

                    User Question
                         │
                         ▼
                  Embedding Model
                         │
                         ▼
                  Query Embedding
                         │
                         ▼
                  L2 Normalization
                         │
                         ▼
                     FAISS Search
                         │
                         ▼
                    Top-K Results
                         │
                         ▼
                 Relevance Threshold
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        Score too low          Relevant chunks
              │                     │
              ▼                     ▼
       No answer found       Build Context
                                    │
                                    ▼
                               Prompt Builder
                                    │
                                    ▼
                               Ollama API
                                    │
                                    ▼
                               gemma3:1b
                                    │
                                    ▼
                           Generated Answer
                                    │
                                    ▼
                             Source Pages