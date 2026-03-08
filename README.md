# RAG Pix Regulation

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

## System Architecture

```
Regulatory PDFs
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
Grounded Answer + Source Citations
       ↓
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
│   ├── vectorstore/      # Weaviate client and indexing
│   ├── retrieval/        # Similarity search
│   ├── rag/              # RAG pipeline orchestration
│   ├── evaluation/       # Retrieval quality, grounding, hallucination metrics
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
| **retrieval** | Similarity search and ranking |
| **rag** | Orchestrate retrieval + generation pipeline |
| **evaluation** | Measure retrieval quality and grounding |
| **observability** | Phoenix integration for tracing and debugging |

---

## Installation Instructions

### Prerequisites

- **Python 3.10+**
- Git

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/rag-pix-regulation.git
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
python scripts/run_ingestion.py   # PDF → corpus_pages.jsonl
python scripts/run_chunking.py   # pages → corpus_chunks.jsonl
python scripts/init_weaviate.py  # Create collection schema
python scripts/run_indexing.py   # Chunks → embeddings → Weaviate
python scripts/demo_retrieval.py # Demo semantic search
```

**Single-command pipeline:**

```bash
python scripts/run_pipeline.py   # Runs ingestion → chunking → init_weaviate → indexing
```

### Data Pipeline

```
PDF (data/raw/)
       ↓
Parsing + Cleaning + Metadata
       ↓
corpus_pages.jsonl
       ↓
Structural segmentation
       ↓
Token chunking
       ↓
corpus_chunks.jsonl
       ↓
Embeddings (BGE-M3)
       ↓
Vector indexing (Weaviate)
       ↓
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
| `article_numbers` | list | Article markers (e.g. ["Art. 1º", "§2º"]) |
| `source_file` | string | Source PDF filename |
| `text` | string | Chunk text content |
| `token_count` | int | Number of tokens |

### Retrieval Evaluation

Evaluate the retriever with Precision@K and Recall@K:

```bash
python scripts/evaluate_retrieval.py
```

Populate `data/evaluation/retrieval_dataset.json` with `relevant_chunks` (chunk_ids) for each query to enable metrics. Run `demo_retrieval.py` to identify relevant chunks, then add their IDs to the dataset.

### Configuration

Place configuration files in `config/`. Parameters such as `embedding_model`, `chunk_size`, `chunk_overlap`, `top_k`, and `llm_model` should be externalized (e.g., `config.yaml`) rather than hardcoded.

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
| **8. Demo & Publication** | Portfolio-ready project | Streamlit app, documentation, video, LinkedIn |

---

## License

This project is licensed under the MIT License — see the LICENSE file for details.