# Evaluation Methodology

This document describes how to evaluate retrieval quality and RAG grounding for the Pix regulation system.

---

## Overview

The evaluation framework measures:

1. **Retrieval** — Precision@K, Recall@K, NDCG@K, MAP@K (page-based relevance)
2. **RAG** — Citation coverage, groundedness, hallucination rate
3. **By difficulty** — Metrics broken down by query difficulty tier

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

### Query Schema

| Field | Description |
|-------|-------------|
| `query` | Natural language question |
| `difficulty` | Difficulty tier (see below) |
| `expected_pages` | Page numbers where the answer is grounded |
| `expected_documents` | Document IDs (e.g. `manual_dict`) |
| `expected_answer_summary` | Ground truth for groundedness |
| `key_concepts` | Checklist for answer completeness |
| `relevant_sources` | Section and pages for reference |

### Difficulty Tiers

| Tier | Description | Purpose |
|------|-------------|---------|
| `single_chunk` | Answer found in a single chunk | Baseline retrieval accuracy |
| `multi_chunk` | Answer requires combining 2+ chunks | Tests context assembly |
| `cross_section` | Answer spans different document sections | Tests broad retrieval |
| `negative` | Answer is NOT in the corpus | Tests abstention / hallucination resistance |

### Topic Coverage

The dataset covers these topic areas:

| Dimension | Query IDs | Topics |
|-----------|-----------|--------|
| `key_management` | eq11-eq19, eq35 | Key types, registration, exclusion, portability, naming rules |
| `process_understanding` | eq1-eq3, eq17, eq20, eq22 | Recovery, portability, possession claims, synchronization |
| `fraud_rules` | eq4, eq6, eq7, eq24, eq25 | Infractions, fraud types, account classifications |
| `procedural_constraints` | eq8, eq9, eq29, eq31 | Deadlines, value limits, cancellation rules |
| `api_and_infrastructure` | eq26-eq28, eq32, eq33 | API limits, cache, events, checkKeys |
| `negative_queries` | neg1-neg4 | Out-of-scope topics (fees, hours, limits, Pix Saque) |

---

## Metrics

### Retrieval Metrics

- **Precision@K** — Fraction of retrieved pages that are relevant
- **Recall@K** — Fraction of relevant pages found (deduplicated to prevent > 1.0)
- **NDCG@K** — Normalized Discounted Cumulative Gain — rewards relevant results at higher positions
- **MAP@K** — Mean Average Precision — combines precision and ranking quality

### RAG Metrics

- **Citation coverage** — Citations match retrieved chunks
- **Groundedness** — Answer uses context, no hallucination heuristics triggered
- **Hallucination rate** — Fraction of responses flagged as suspicious

---

## Workflow

### 1. Retrieval-only evaluation

Measures how well the retriever finds relevant pages.

```bash
python scripts/evaluate_retrieval.py
```

### 2. Full RAG evaluation

Measures retrieval + generation quality.

```bash
python scripts/evaluate_rag.py
```

### 3. View report

```bash
cat reports/evaluation_report.json
```

The report includes:
- Aggregated retrieval metrics
- Aggregated RAG metrics
- **`by_difficulty`** — Metrics broken down by difficulty tier
- Per-query results with difficulty labels

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
   - `query`, `difficulty`, `expected_pages`, `expected_documents`
   - Optional: `expected_answer_summary`, `key_concepts`, `relevant_sources`
3. Choose the appropriate difficulty tier based on expected answer complexity
