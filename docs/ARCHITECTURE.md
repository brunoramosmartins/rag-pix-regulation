# RAG Pix Regulation — Architecture

This document describes the modular architecture for the RAG system that serves fraud prevention analysts, compliance teams, and AI agents with accurate, traceable answers grounded in Brazilian Pix regulatory documents (BCB resolutions, MED 2.0).

---

## System Flow

```mermaid
flowchart TD
    A[Regulatory PDFs<br/>BCB · MED 2.0] --> B[Ingestion<br/>pdf_loader · text_cleaner · metadata_extractor]
    B --> C[corpus_pages.jsonl]
    C --> D[Chunking<br/>structural_segmenter → token_chunker]
    D --> E[corpus_chunks.jsonl]
    E --> F[Embeddings<br/>BAAI/bge-m3]
    F --> G[Weaviate<br/>Vector Database]

    H[User Query] --> I[Query Embedding<br/>BAAI/bge-m3]
    I --> J[Vector Search<br/>ANN · top-k]
    G --> J
    J --> K[Context Builder<br/>token budget · citation markers]
    K --> L[Prompt Template<br/>system instruction + context + query]
    L --> M[LLM<br/>Llama-3.2-3B via Ollama]
    M --> N[Answer + Citations]

    N --> O[Evaluation<br/>Precision@K · Recall@K · Grounding]
    N --> P[Phoenix Tracing<br/>OpenTelemetry spans]
```

---

## Tech Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **LLM** | meta-llama/Llama-3.2-3B-Instruct via Ollama | Local execution, no API cost, deterministic (temp=0) |
| **Embeddings** | BAAI/bge-m3 | State-of-the-art multilingual model; strong Portuguese performance; 1024-dim dense vectors |
| **Vector DB** | Weaviate | Production-grade, cosine similarity native support, schema validation, batch insert API |
| **Observability** | Phoenix + OpenTelemetry | RAG-native span kinds (retriever, chain, LLM), no vendor lock-in |
| **Application** | Streamlit | Rapid prototyping, side-by-side baseline vs. RAG comparison |
| **Serialization** | JSONL + Pydantic v2 | Line-delimited streaming, schema enforcement, zero serialization overhead |

---

## Architecture Decision Records

### ADR-01: BGE-M3 over OpenAI Embeddings

**Context:** The corpus is in Portuguese (legal/technical register). Embedding quality for this domain is critical.

**Decision:** Use `BAAI/bge-m3` (1024-dim, multilingual) instead of OpenAI's `text-embedding-ada-002` or `text-embedding-3-small`.

**Rationale:**
- BGE-M3 achieves state-of-the-art results on Portuguese MTEB benchmarks
- Runs locally with no API cost or rate limits
- Deterministic: same input always produces the same vector
- 1024 dimensions provide rich semantic representation for regulatory text

**Trade-off:** Requires ~1.3 GB local model download on first use. Acceptable for a portfolio project; mitigated by model caching.

---

### ADR-02: Structural Segmentation Before Token Chunking

**Context:** Raw PDF pages don't respect semantic boundaries. Naive fixed-size chunking splits articles mid-sentence.

**Decision:** Apply two-stage chunking:
1. **Structural segmentation** — split pages at article/paragraph markers (`Art.`, `§`, numbered sections)
2. **Token chunking** — apply sliding window (500 tokens, 50 overlap) within each segment

**Rationale:**
- Regulatory text has natural semantic units (articles, paragraphs, sections)
- Structural segmentation preserves legal coherence — a chunk about "Art. 5º" doesn't bleed into "Art. 6º"
- Token chunking ensures each chunk fits the model's context window
- Overlap retains continuity across chunk boundaries

**Trade-off:** Two-pass pipeline is more complex than naive chunking. Complexity is bounded and well-tested.

---

### ADR-03: Weaviate over Chroma or FAISS

**Context:** Need a vector database for embedding storage and ANN retrieval.

**Decision:** Use Weaviate (Docker, local) instead of Chroma (embedded) or FAISS (in-memory).

**Rationale:**
- Weaviate persists data across runs without reimporting; Chroma is embedded (reset risk); FAISS is in-memory only
- Schema validation ensures consistent property types at insert time
- Batch import API enables efficient bulk indexing
- Docker Compose deployment mirrors production patterns better than an embedded DB

**Trade-off:** Requires Docker. Mitigated by providing `docker-compose.yml` with a single command start.

---

### ADR-04: Ollama for Local LLM Inference

**Context:** Need a text-generation LLM for the RAG answer step without API keys or costs.

**Decision:** Use Ollama serving `llama3.2:3b` locally.

**Rationale:**
- Zero API cost for development and evaluation
- Deterministic inference with `temperature=0, top_p=1.0`
- Model abstracted behind `LLMClient` interface — trivial to swap for OpenAI, Anthropic, or any other provider
- 3B parameter model runs on CPU; fits portfolio constraints

**Trade-off:** Quality ceiling lower than GPT-4 class models. Acceptable for a retrieval-grounded system where the LLM role is primarily formatting and synthesis.

---

## Module Responsibilities

| Module | Key Files | Inputs | Outputs |
|--------|-----------|--------|---------|
| **ingestion** | `pdf_loader`, `text_cleaner`, `metadata_extractor`, `serializer` | PDF files | `corpus_pages.jsonl` |
| **chunking** | `structural_segmenter`, `token_chunker`, `serializer`, `loader` | `corpus_pages.jsonl` | `corpus_chunks.jsonl` |
| **embeddings** | `embedding_generator`, `validation` | `corpus_chunks.jsonl` | `(Chunk, vector)` pairs |
| **vectorstore** | `weaviate_client`, `indexer` | `(Chunk, vector)` pairs | Weaviate collection |
| **retrieval** | `query_embedding`, `vector_search`, `retriever` | User query string | `list[RetrievalResult]` |
| **rag** | `context_builder`, `prompt_template`, `rag_pipeline` | `list[RetrievalResult]` + LLM | `RAGResponse` with citations |
| **evaluation** | `dataset_loader`, `retrieval_metrics`, `rag_evaluation`, `evaluation_runner` | Dataset + RAG fn | JSON evaluation report |
| **observability** | `tracing` | Span names + attributes | OpenTelemetry spans → Phoenix |
| **demo** | `demo_service` | Query string | Baseline + RAG result dicts |

---

## Inter-Module Data Flow

```
PDF files
  → [ingestion] → Page (Pydantic model)
  → [serializer] → JSONL line (corpus_pages.jsonl)

corpus_pages.jsonl
  → [chunking.structural_segmenter] → StructuralSegment (Pydantic model)
  → [chunking.token_chunker] → Chunk (Pydantic model)
  → [chunking.serializer] → JSONL line (corpus_chunks.jsonl)

corpus_chunks.jsonl
  → [embeddings.embedding_generator] → (Chunk, list[float])
  → [vectorstore.indexer] → Weaviate object

User query: str
  → [retrieval.query_embedding] → list[float]  (1024-dim)
  → [retrieval.vector_search] → list[dict]
  → [retrieval.retriever] → list[RetrievalResult]
  → [rag.context_builder] → str  (token-budgeted context)
  → [rag.prompt_template] → str  (final prompt)
  → [llm.baseline_llm] → (answer: str, usage: LLMUsage)
  → [rag.rag_pipeline] → RAGResponse
```

---

## Repository Structure

```
rag-pix-regulation/
├── data/
│   ├── raw/              # Original regulatory documents (PDFs)
│   ├── evaluation/       # Retrieval evaluation dataset (ground truth)
│   └── processed/        # Chunked documents, embeddings
├── src/
│   ├── ingestion/        # Document loading and PDF parsing
│   ├── chunking/         # Document splitting strategies
│   ├── embeddings/       # Vector representations
│   ├── vectorstore/      # Weaviate client and indexing
│   ├── retrieval/        # Similarity search
│   ├── rag/              # RAG pipeline orchestration
│   ├── evaluation/       # Retrieval quality, grounding, hallucination metrics
│   ├── observability/    # Phoenix tracing and monitoring
│   ├── demo/             # Service layer for Streamlit UI
│   └── utils/            # System health checks
├── scripts/              # Runnable pipeline and validation scripts
├── tests/                # Unit and integration tests
├── config/               # Configuration (YAML, env templates)
├── docs/                 # Architecture and design docs
└── app/                  # Streamlit application
```

---

## Design Principles

- **Separation of concerns** — Each module has a single responsibility with no circular dependencies
- **Reproducibility** — Config-driven parameters, versioned data (JSONL), deterministic embeddings
- **Extensibility** — `LLMClient` abstract interface allows swapping inference backends; vector store abstracted behind `weaviate_client`
- **Testability** — Unit tests per module; integration tests skip gracefully when services are unavailable
- **Traceability** — Every answer cites source document and page; chunk IDs are deterministic (`{doc_id}_p{page}_s{segment}_c{chunk}`)
- **Observability** — Phoenix tracing wraps all RAG spans (retriever → context → prompt → LLM) with input/output attributes

---

## Architecture Diagrams

Detailed Mermaid diagrams are available in `docs/architecture/`:

| Diagram | Description |
|---------|-------------|
| [system_overview.md](architecture/system_overview.md) | High-level flow: query → embed → search → context → prompt → LLM |
| [ingestion_pipeline.md](architecture/ingestion_pipeline.md) | PDF → parsing → corpus_pages.jsonl |
| [chunking_pipeline.md](architecture/chunking_pipeline.md) | Pages → structural segmentation → token chunking → corpus_chunks.jsonl |
| [embedding_indexing_pipeline.md](architecture/embedding_indexing_pipeline.md) | Chunks → BGE-M3 → Weaviate |
| [retrieval_rag_pipeline.md](architecture/retrieval_rag_pipeline.md) | Retrieval + RAG flow with citations |

---

## Demo Service Layer

The interactive demo (`app/streamlit_app.py`) uses a service layer (`src/demo/demo_service.py`) to:

- Execute baseline queries (LLM without retrieval)
- Execute RAG queries (retrieve → context → prompt → LLM)
- Perform health checks (Weaviate, Ollama)
- Return standardized response objects for the UI

This keeps the UI decoupled from the pipeline implementation.
