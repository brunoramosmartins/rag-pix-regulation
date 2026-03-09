# Evaluation Datasets

## Retrieval Evaluation (`retrieval_dataset.json`)

To enable Precision@K and Recall@K metrics, populate `retrieval_dataset.json` with ground truth.

## RAG Evaluation (`rag_queries.json`)

For RAG context grounding: does retrieved context contain the expected documents?

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
```

Run evaluation:
```bash
python scripts/evaluate_rag.py
```

## Workflow

1. **Index your corpus**
   ```bash
   python scripts/run_pipeline.py
   ```

2. **Run retrieval demo** to see returned chunks
   ```bash
   python scripts/demo_retrieval.py
   ```

3. **Identify relevant chunks** for each query from the output (inspect `chunk_id` values)

4. **Add chunk IDs** to `retrieval_dataset.json` in the `relevant_chunks` array for each query

## Example

```json
{
  "query_id": "q1",
  "query": "Como funciona o cadastro de chave Pix?",
  "relevant_chunks": ["manual_dict_p5_s0_c0", "manual_dict_p6_s1_c0"]
}
```

Chunk IDs follow the format: `{document_id}_p{page}_s{segment}_c{chunk}`.
