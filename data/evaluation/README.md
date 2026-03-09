# Evaluation Datasets

## Canonical Dataset (`rag_evaluation_dataset.json`)

Single source for retrieval and RAG evaluation — Manual Operacional do DICT.

Each query includes:

- `query` — natural language question
- `expected_pages` — page numbers where answer is grounded (for retrieval metrics)
- `expected_documents` — document IDs (e.g. manual_dict)
- `expected_answer_summary` — ground truth for groundedness
- `key_concepts` — checklist for answer completeness
- `relevant_sources` — section and pages for reference

## Workflow

1. **Index your corpus**
   ```bash
   python scripts/run_pipeline.py
   ```

2. **Run retrieval evaluation**
   ```bash
   python scripts/evaluate_retrieval.py
   ```

3. **Run full RAG evaluation** (requires Ollama)
   ```bash
   python scripts/evaluate_rag.py
   ```

4. **View report**
   ```bash
   cat reports/evaluation_report.json
   ```

## Metrics

- **Retrieval:** Precision@K, Recall@K (page-based relevance)
- **RAG:** citation coverage, groundedness, hallucination rate
