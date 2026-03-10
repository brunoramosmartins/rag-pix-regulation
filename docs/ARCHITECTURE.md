# RAG Pix Regulation — Architecture

This document describes the modular architecture for the RAG system that serves fraud prevention analysts, compliance teams, and AI agents with accurate, traceable answers grounded in Brazilian Pix regulatory documents (BCB resolutions, MED 2.0).

---

## System Flow

```
Regulatory PDFs (BCB, MED 2.0)
            ↓
       Parsing
            ↓
       Chunking
            ↓
      Embeddings
            ↓
Vector Database (Weaviate)
            ↓
       Retriever
            ↓
    Prompt Builder
            ↓
           LLM
            ↓
  Response + Citations
            ↓
Evaluation + Observability (Phoenix)
```

---

## Tech Stack

| Layer | Technology |
|-------|-------------|
| **LLM** | meta-llama/Llama-3.2-3B-Instruct |
| **Embeddings** | BAAI/bge-m3 (multilingual, Portuguese) |
| **Vector DB** | Weaviate (embedded) |
| **Observability** | Phoenix + OpenTelemetry |
| **Application** | Streamlit |

---

## Repository Structure

```
rag-pix-regulation/
├── data/
│   ├── raw/              # Original regulatory documents (PDFs)
│   └── processed/        # Chunked documents, embeddings
├── src/
│   ├── ingestion/        # Document loading and PDF parsing
│   ├── chunking/         # Document splitting strategies
│   ├── embeddings/       # Vector representations
│   ├── retrieval/        # Similarity search
│   ├── rag/              # RAG pipeline orchestration
│   ├── evaluation/       # Retrieval quality, grounding, hallucination metrics
│   └── observability/    # Phoenix tracing and monitoring
├── scripts/              # Utility and pipeline scripts
├── notebooks/            # Experimentation and analysis
├── tests/                # Unit and integration tests
├── config/               # Configuration (YAML, env templates)
├── docs/                 # Project documentation
└── app/                  # Streamlit application
```

---

## Module Responsibilities

| Module | Purpose |
|--------|---------|
| **ingestion** | Load documents from sources, parse PDFs, extract text and metadata |
| **chunking** | Split documents into retrieval-optimized chunks with overlap |
| **embeddings** | Generate vector representations for semantic search |
| **retrieval** | Similarity search, ranking, top-k retrieval |
| **rag** | Orchestrate retrieval + generation, prompt building, citation injection |
| **evaluation** | Measure Recall@K, Precision@K, grounding, hallucination rates |
| **observability** | Phoenix integration for tracing, token usage, context inspection |

---

## Design Principles

- **Separation of concerns**: Each module has a single responsibility
- **Reproducibility**: Config-driven, versioned data and models
- **Extensibility**: Modular design allows swapping components (embedding models, vector stores, LLMs)
- **Testability**: Unit tests per module, integration tests for pipelines
- **Traceability**: Responses cite source documents for audit and compliance
- **Observability**: Phoenix tracing for debugging and performance analysis

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
