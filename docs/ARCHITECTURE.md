# RAG Pix Regulation - Architecture

## Repository Structure

This document describes the modular architecture for the RAG system that interprets Brazilian Pix regulatory documents.

```
rag-pix-regulation/
├── data/                 # Datasets
│   ├── raw/              # Original documents (PDFs, etc.)
│   └── processed/        # Chunked documents, embeddings
├── src/                  # Core modules
│   ├── ingestion/        # Document loading and preprocessing
│   ├── chunking/         # Document splitting strategies
│   ├── embeddings/       # Vector representations
│   ├── retrieval/        # Similarity search
│   ├── rag/              # RAG pipeline orchestration
│   ├── evaluation/       # Quality metrics (retrieval, grounding, hallucination)
│   └── observability/    # Phoenix tracing and monitoring
├── scripts/              # Utility and pipeline scripts
├── notebooks/            # Jupyter notebooks for experimentation
├── tests/                # Test suite
├── config/               # Configuration files
├── docs/                 # Project documentation
└── app/                  # Application layer (API, CLI, web)
```

## Module Responsibilities

| Module | Purpose |
|--------|---------|
| **ingestion** | Load documents from various sources, parse formats (PDF, etc.) |
| **chunking** | Split documents into chunks suitable for embedding and retrieval |
| **embeddings** | Generate vector representations for semantic search |
| **retrieval** | Similarity search, ranking, and document retrieval |
| **rag** | Orchestrate retrieval + generation pipeline |
| **evaluation** | Measure retrieval quality, grounding, hallucination rates |
| **observability** | Phoenix integration for tracing and monitoring |

## Design Principles

- **Separation of concerns**: Each module has a single responsibility
- **Reproducibility**: Config-driven, versioned data and models
- **Extensibility**: Modular design allows swapping components (e.g., embedding models, vector stores)
- **Testability**: Unit tests per module, integration tests for pipelines
