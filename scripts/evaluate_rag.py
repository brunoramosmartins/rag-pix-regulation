#!/usr/bin/env python
"""
Evaluate RAG context grounding: does retrieved context contain expected documents?

Uses data/evaluation/rag_queries.json with expected_documents per query.

Run from project root (Weaviate required):

    python scripts/evaluate_rag.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import retrieve
from src.vectorstore.weaviate_client import is_weaviate_ready

RAG_QUERIES_PATH = PROJECT_ROOT / "data" / "evaluation" / "rag_queries.json"


def load_rag_queries(path: Path) -> list[dict]:
    """Load RAG evaluation queries."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("queries", [])


def evaluate_context_grounding(
    query: str,
    expected_documents: list[str],
    top_k: int = 5,
) -> tuple[bool, set[str]]:
    """
    Check if retrieved chunks contain at least one expected document.

    Returns (grounded, retrieved_doc_ids).
    """
    results = retrieve(query, top_k=top_k)
    retrieved_ids = {r.document_id for r in results}
    expected = set(expected_documents)
    grounded = bool(retrieved_ids & expected)
    return grounded, retrieved_ids


def main() -> None:
    """Run RAG context grounding evaluation."""
    if not is_weaviate_ready():
        print("ERROR: Weaviate is not running. Run: python scripts/run_indexing.py")
        sys.exit(1)

    if not RAG_QUERIES_PATH.exists():
        print(f"ERROR: {RAG_QUERIES_PATH} not found")
        sys.exit(1)

    queries = load_rag_queries(RAG_QUERIES_PATH)
    if not queries:
        print("No queries in rag_queries.json")
        sys.exit(0)

    grounded_count = 0
    for q in queries:
        query = q.get("query", "")
        expected = q.get("expected_documents", [])
        if not query or not expected:
            continue
        grounded, retrieved = evaluate_context_grounding(query, expected)
        if grounded:
            grounded_count += 1
        status = "OK" if grounded else "MISS"
        print(f"  [{status}] {q.get('query_id', '?')}: {query[:50]}...")
        print(f"       Expected: {expected}, Retrieved: {retrieved}")

    total = len([q for q in queries if q.get("expected_documents")])
    if total:
        print(f"\nContext grounding: {grounded_count}/{total} queries")


if __name__ == "__main__":
    main()
