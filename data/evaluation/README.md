# Evaluation Datasets

This project includes multiple datasets used to evaluate retrieval quality and RAG response groundedness.

---

## Retrieval Evaluation (`retrieval_dataset.json`)

Dataset used to evaluate **retrieval performance**.

Each query maps to the relevant document chunks that should be returned by the vector search system.

Example schema:

{
  "queries": [
    {
      "query_id": "q1",
      "query": "How does Pix key registration work?",
      "relevant_chunk_ids": ["doc_p5_s2_c1"]
    }
  ]
}

Used for computing:

- Precision@K
- Recall@K

---

## Canonical Dataset (`rag_evaluation_dataset.json`)

Primary dataset used for **combined retrieval and RAG evaluation**.

Source: *Manual Operacional do DICT*.

Each query includes:

- `query` — natural language question
- `expected_pages` — page numbers where the answer is grounded
- `expected_documents` — document identifiers (e.g. `manual_dict`)
- `expected_answer_summary` — reference summary of the expected answer
- `key_concepts` — checklist of concepts that must appear in the answer
- `relevant_sources` — reference sections and pages

This dataset is used by the full evaluation pipeline.

---

## RAG Evaluation (`rag_queries.json`)

Dataset used for **context grounding validation**.

The evaluation verifies whether retrieved context contains the expected regulatory documents.

Structure:

```json
{
  "queries": [
    {
      "query_id": "rq1",
      "query": "Como funciona o cadastro de chave Pix?",
      "expected_documents": ["manual_dict"]
    }
  ]
}