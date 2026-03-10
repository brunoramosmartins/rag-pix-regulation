# Evaluation Methodology

This document describes how to evaluate retrieval quality and RAG grounding for the Pix regulation system.

---

## Overview

The evaluation framework measures:

1. **Retrieval** — Precision@K, Recall@K (page-based relevance)
2. **RAG** — Citation coverage, groundedness, hallucination rate

---

## Prerequisites

1. **Indexed corpus**
   ```bash
   python scripts/run_pipeline.py
   ```

2. **Weaviate** running
   ```bash
   docker compose up -d
   ```

3. **Ollama** (for full RAG evaluation)
   ```bash
   ollama pull llama3.2:3b
   ollama serve
   ```

---

## Dataset

**Location:** `data/evaluation/rag_evaluation_dataset.json`

Each query includes:

| Field | Description |
|-------|-------------|
| `query` | Natural language question |
| `expected_pages` | Page numbers where the answer is grounded |
| `expected_documents` | Document IDs (e.g. `manual_dict`) |
| `expected_answer_summary` | Ground truth for groundedness |
| `key_concepts` | Checklist for answer completeness |
| `relevant_sources` | Section and pages for reference |

---

## Workflow

### 1. Retrieval-only evaluation

Measures how well the retriever finds relevant pages.

```bash
python scripts/evaluate_retrieval.py
```

**Metrics:**
- **Precision@K** — Fraction of retrieved pages that are relevant
- **Recall@K** — Fraction of relevant pages that were retrieved

### 2. Full RAG evaluation

Measures retrieval + generation quality.

```bash
python scripts/evaluate_rag.py
```

**Metrics:**
- **Citation coverage** — Citations match retrieved chunks
- **Groundedness** — Answer uses context, no hallucination heuristics triggered
- **Hallucination rate** — Fraction of responses flagged as suspicious

### 3. View report

```bash
cat reports/evaluation_report.json
```

---

## Phoenix Tracing (optional)

For trace visualization during evaluation:

1. Start Phoenix:
   ```bash
   python -m phoenix.server.main serve
   ```

2. Run evaluation (traces sent to `http://localhost:6006`):
   ```bash
   python scripts/evaluate_rag.py
   ```

3. Open Phoenix UI to inspect spans (retrieval, context, prompt, LLM).

---

## Adding New Evaluation Queries

1. Identify relevant chunks via `python scripts/demo_retrieval.py`
2. Add query to `data/evaluation/rag_evaluation_dataset.json` with:
   - `query`, `expected_pages`, `expected_documents`
   - Optional: `expected_answer_summary`, `key_concepts`, `relevant_sources`
