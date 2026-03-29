# RAG Pix Regulation

[![CI](https://github.com/brunoramosmartins/rag-pix-regulation/actions/workflows/ci.yml/badge.svg)](https://github.com/brunoramosmartins/rag-pix-regulation/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Weaviate](https://img.shields.io/badge/Vector%20DB-Weaviate-orange.svg)](https://weaviate.io/)
[![OpenTelemetry](https://img.shields.io/badge/Observability-OpenTelemetry%20%2B%20Phoenix-purple.svg)](https://docs.arize.com/phoenix)

A **Retrieval-Augmented Generation (RAG)** system for querying Brazilian Pix regulatory documentation. Built for fraud prevention teams, compliance analysts, and AI agents that require accurate, traceable answers grounded in official Central Bank (BCB) regulations and the DICT Operational Manual (MED 2.0).

---

## Project Overview

### Problem

Fraud prevention analysts, compliance officers, legal teams, and support engineers routinely need to consult Pix regulations—including BCB resolutions and the MED 2.0 manual. Manual PDF search is slow, error-prone, and scales poorly. When analysts or AI agents rely on generic LLMs without regulatory context, responses can be inaccurate or hallucinated, creating compliance and operational risk.

### Solution

This RAG system indexes official Pix documentation, retrieves only relevant excerpts, and injects them into the LLM prompt. The result:

- **Faster regulatory lookup** — Analysts get answers in seconds instead of scanning hundreds of pages
- **Higher accuracy** — Responses are grounded exclusively in indexed documents
- **Traceable citations** — Every answer links back to specific regulatory sources
- **Agent-ready** — Fraud prevention agents can use the RAG for up-to-date, auditable information

### Target Users

| User | Use Case |
|------|----------|
| **Fraud analysts** | Query rules, validate scenarios, review policies |
| **AI agents** | Retrieve current regulations for automated fraud detection workflows |
| **Compliance teams** | Verify interpretations, prepare reports, audit decisions |
| **Support & engineering** | Resolve operational questions with authoritative sources |

### Success Criteria

The project is successful when:

- The system answers regulatory questions with high precision
- Responses are grounded in retrieved excerpts with citations
- Retrieval metrics (Recall@K, Precision@K) meet defined thresholds
- The repository demonstrates professional engineering practices
- The project is suitable as a technical portfolio piece

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| **1. Setup** | Repo, venv, dependencies | Done |
| **2. Ingestion** | PDF parsing, text extraction, metadata | Done |
| **3. Chunking** | Structural + token-based chunking | Done |
| **4. Embeddings** | BGE-M3 vectors, Weaviate indexing | Done |
| **5. Retrieval** | Semantic search, Recall@K, Precision@K | Done |
| **6. RAG Pipeline** | Prompt template, answer generation, citations | Done |
| **7. Evaluation & Observability** | Metrics, Phoenix tracing, evaluation runner | Done |
| **8. Incremental Indexing** | Content hashing, change detection, ingestion tracing | Done |
| **9. Demo & Publication** | Streamlit app, documentation | In Progress |

---

## Recent: Incremental Indexing & Ingestion Observability (v1.1.0)

Inspired by [Two years of vector search at Notion](https://www.notion.com/pt/blog/two-years-of-vector-search-at-notion), the indexing pipeline was upgraded from full-reindex to **incremental indexing** with content-hash change detection. This addresses the article's core insight: **the dominant cost in RAG systems is not search, but maintaining and updating embeddings.**

### What changed

| Before | After |
|--------|-------|
| Every run re-embeds all chunks | Only new/changed chunks are embedded |
| No change detection | SHA-256 content hash per chunk |
| No embedding provenance | `embedding_model` stored per vector |
| Ingestion pipeline untraced | Full OpenTelemetry tracing on ingestion |

### How it works

```
Chunks loaded from JSONL
        |
Fetch existing hashes from Weaviate
        |
Classify: new / changed / unchanged
        |
Skip unchanged --- only embed new + changed
        |
Insert with content_hash + embedding_model metadata
        |
Log: "Indexed 12 new, 3 updated, 85 skipped"
```

### Impact assessment (Notion article practices)

| Practice | Status | Notes |
|----------|--------|-------|
| Content hash change detection | Implemented | SHA-256 per chunk |
| Incremental indexing | Implemented | Skip unchanged, update changed |
| Embedding versioning metadata | Implemented | `embedding_model` stored per vector |
| Ingestion pipeline tracing | Implemented | Full spans in Phoenix |
| Decoupled ingestion/serving | Already done | Stateless serving reads from Weaviate |
| Self-hosted embeddings | Already done | BGE-M3 via sentence-transformers |
| Batch processing | Already done | batch_size=32 embeddings |
| Sharding / real-time ingestion | Skipped | Not relevant at current scale |

---

## System Architecture

```
Regulatory PDFs
       |
   Parsing
       |
   Chunking (+ SHA-256 content hash)
       |
  Embeddings
       |
Vector Database (Weaviate)
  [content_hash + embedding_model metadata]
       |
   Retriever
       |
  Prompt Builder
       |
      LLM
       |
Grounded Answer + Source Citations
       |
Evaluation + Observability (Phoenix)
```

### Tech Stack

| Layer | Technology |
|------|------------|
| **LLM** | meta-llama/Llama-3.2-3B-Instruct |
| **Embeddings** | BAAI/bge-m3 |
| **Vector Database** | Weaviate |
| **Observability** | Phoenix + OpenTelemetry |
| **Application** | Streamlit |
| **Core Libraries** | sentence-transformers, weaviate-client, pydantic |

---

## Repository Structure

```
rag-pix-regulation/
├── data/
│   ├── raw/              # Original regulatory documents (PDFs)
│   ├── evaluation/       # Retrieval evaluation dataset
│   └── processed/        # Chunked documents, embeddings
├── src/
│   ├── ingestion/        # Document loading and PDF parsing
│   ├── chunking/         # Document splitting strategies
│   ├── embeddings/       # Vector representations
│   ├── vectorstore/      # Weaviate client, schema, incremental indexing
│   ├── retrieval/        # Similarity search
│   ├── rag/              # RAG pipeline orchestration
│   ├── evaluation/       # Retrieval quality, grounding, hallucination metrics
│   ├── config/           # Pydantic Settings, structured logging
│   └── observability/    # Phoenix tracing and monitoring
├── scripts/              # Utility and pipeline scripts
├── notebooks/            # Experimentation and analysis
├── tests/                # Unit and integration tests
├── config/               # Configuration (YAML, env templates)
├── docs/                 # Architecture and design docs
└── app/                  # Streamlit application
```

| Module | Purpose |
|--------|---------|
| **ingestion** | Load and parse documents from various sources |
| **chunking** | Split documents into retrieval-optimized chunks |
| **embeddings** | Generate vector representations for semantic search |
| **vectorstore** | Weaviate client, schema management, incremental indexing |
| **retrieval** | Similarity search and ranking |
| **rag** | Orchestrate retrieval + generation pipeline |
| **evaluation** | Measure retrieval quality and grounding |
| **config** | Centralized Pydantic Settings with YAML + env var support |
| **observability** | Phoenix integration for tracing and debugging |

---

## Installation Instructions

### Prerequisites

- **Python 3.10+**
- Git

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/brunoramosmartins/rag-pix-regulation.git
   cd rag-pix-regulation
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # or
   .venv\Scripts\activate     # Windows (Git Bash / PowerShell)
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**

   ```bash
   pip list
   python -c "import sentence_transformers; import weaviate; import phoenix; import streamlit; print('OK')"
   ```

### Weaviate (Vector Database)

For embedding and retrieval, run Weaviate locally:

```bash
docker compose up -d
```

Then run the full pipeline (or use the single script below):

```bash
python scripts/run_ingestion.py   # PDF -> corpus_pages.jsonl
python scripts/run_chunking.py   # pages -> corpus_chunks.jsonl
python scripts/init_weaviate.py  # Create collection schema
python scripts/run_indexing.py   # Chunks -> embeddings -> Weaviate
python scripts/demo_retrieval.py # Demo semantic search
```

**Single-command pipeline:**

```bash
python scripts/run_pipeline.py   # Runs ingestion -> chunking -> init_weaviate -> indexing
```

**Incremental re-indexing:** After the first run, subsequent runs only process new or changed chunks:

```bash
python scripts/run_indexing.py   # Skips unchanged chunks automatically
# Output: "Indexed 0 new, 0 updated, 120 skipped"
```

### Data Pipeline

```
PDF (data/raw/)
       |
Parsing + Cleaning + Metadata
       |
corpus_pages.jsonl
       |
Structural segmentation
       |
Token chunking (+ SHA-256 content hash)
       |
corpus_chunks.jsonl
       |
Incremental indexing (skip unchanged)
       |
Embeddings (BGE-M3)
       |
Vector indexing (Weaviate)
  [content_hash + embedding_model metadata]
       |
Semantic retrieval
```

### Dataset Format

The chunk dataset (`corpus_chunks.jsonl`) uses one JSON object per line:

| Field | Type | Description |
|-------|------|--------------|
| `chunk_id` | string | Deterministic ID: `{document_id}_p{page}_s{segment}_c{chunk}` |
| `document_id` | string | Source document identifier |
| `page_number` | int | 1-based page index |
| `segment_index` | int | Segment index within page |
| `chunk_index` | int | Chunk index within segment |
| `section_title` | string | Section title (e.g. "1 Chaves Pix") |
| `article_numbers` | list | Article markers (e.g. ["Art. 1o", "par.2o"]) |
| `source_file` | string | Source PDF filename |
| `text` | string | Chunk text content |
| `token_count` | int | Number of tokens |
| `content_hash` | string | SHA-256 hash of text for change detection |

### Demo (Interactive Interface)

Run the Streamlit demo to compare **Baseline LLM** (no retrieval) vs **RAG Pipeline** (with retrieval):

```bash
streamlit run app/streamlit_app.py
```

**Prerequisites:** Weaviate running, pipeline indexed, Ollama with `llama3.2:3b` pulled.

**Example queries:**
- Como funciona o registro de chave Pix?
- Quais sao as regras de devolucao por fraude?
- Como funciona a portabilidade de chave Pix?
- Quantas chaves Pix posso ter por conta?
- O que e o DICT no contexto do Pix?

The demo shows side-by-side responses, citations, and retrieved context chunks.

**Optional — Phoenix traces:** To visualize RAG chain spans (retrieval, context, prompt, LLM) in Phoenix, start it in another terminal before launching the demo:

```bash
python -m phoenix.server.main serve
```

Then open http://localhost:6006 to inspect traces. The demo registers with Phoenix automatically when it is available.

### Evaluation

See [docs/EVALUATION.md](docs/EVALUATION.md) for the full evaluation workflow.

**Retrieval-only:**
```bash
python scripts/evaluate_retrieval.py
```

**Full RAG (requires Ollama):**
```bash
python scripts/evaluate_rag.py
```

**Report:** `reports/evaluation_report.json`

### Configuration

Default parameters are defined in [`config/config.yaml`](config/config.yaml). Copy `.env.example` to `.env` for environment-specific overrides.

Key parameters:

| Parameter | Default | File |
|-----------|---------|------|
| `embedding_model` | `BAAI/bge-m3` | `config/config.yaml` |
| `chunk_size` | `500` | `config/config.yaml` |
| `chunk_overlap` | `50` | `config/config.yaml` |
| `top_k` | `5` | `config/config.yaml` |
| `llm_model` | `llama3.2:3b` | `config/config.yaml` |

---

## Roadmap

| Phase | Objective | Deliverables |
|-------|-----------|--------------|
| **1. Setup** | Reproducible environment and project structure | Repo, venv, dependencies, README |
| **2. Ingestion** | Structured regulatory corpus | PDF parsing, text extraction, metadata |
| **3. Chunking** | Semantically coherent chunks | Chunking strategy, overlap, metadata preservation |
| **4. Embeddings** | Vector representation of corpus | Embedding pipeline, Weaviate indexing |
| **5. Retrieval** | Validated retriever | Semantic search, Recall@K, Precision@K |
| **6. RAG Pipeline** | End-to-end question answering | Prompt template, RAG pipeline, baseline comparison |
| **7. Evaluation & Observability** | Measurable, observable system | Metrics, Phoenix tracing, logs |
| **8. Incremental Indexing** | Production-grade indexing pipeline | Content hashing, change detection, ingestion tracing |
| **9. Demo & Publication** | Portfolio-ready project | Streamlit app, documentation, video, LinkedIn |

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and module responsibilities |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Evaluation methodology and workflow |
| [docs/architecture/](docs/architecture/) | Mermaid diagrams (ingestion, chunking, embedding, retrieval, RAG) |

---

## License

This project is licensed under the MIT License — see the LICENSE file for details.
