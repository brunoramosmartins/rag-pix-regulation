#!/usr/bin/env python
"""
Demo: semantic search over indexed regulatory chunks.

Run from project root (after run_indexing.py):

    python scripts/demo_retrieval.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import retrieve  # noqa: E402
from src.vectorstore.weaviate_client import is_weaviate_ready  # noqa: E402

DEMO_QUERIES = [
    "Como funciona o cadastro de chave Pix?",
    "Regras para devolução por fraude",
    "Processo de portabilidade de chave Pix",
]


def main() -> None:
    """Run demo queries and display results with similarity scores."""
    if not is_weaviate_ready():
        print("ERROR: Weaviate is not running. Run: python scripts/run_indexing.py")
        sys.exit(1)

    for query in DEMO_QUERIES:
        print(f"\n--- Query: {query} ---\n")
        results = retrieve(query, top_k=3)
        for i, r in enumerate(results, 1):
            score_str = f" (similarity: {r.similarity_score})" if r.similarity_score is not None else ""
            print(f"--- Result {i}{score_str} ---")
            print(f"document_id: {r.document_id}")
            print(f"page_number: {r.page_number}")
            print(f"section_title: {r.section_title}")
            print(f"text (first 200 chars): {r.text[:200]}...")
            print()


if __name__ == "__main__":
    main()
