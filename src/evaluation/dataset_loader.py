"""Unified evaluation dataset loader — single source for retrieval and RAG evaluation."""

import json
from pathlib import Path
from typing import Any


def load_evaluation_dataset(path: Path) -> dict[str, Any]:
    """
    Load canonical evaluation dataset from JSON.

    Expected format (rag_evaluation_dataset.json):
    {
      "queries": [
        {
          "query_id": "eq1",
          "query": "...",
          "expected_pages": [122],
          "expected_documents": ["manual_dict"],
          "expected_answer_summary": "...",
          "key_concepts": [...]
        }
      ]
    }
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "queries" not in data:
        raise ValueError("Invalid evaluation dataset: missing 'queries' field")

    for i, q in enumerate(data["queries"]):
        if "query" not in q:
            raise ValueError(f"Invalid query entry at index {i}: missing 'query' field")

    return data


def get_expected_pages(query: dict[str, Any]) -> set[int]:
    """Extract expected page numbers from query. Supports expected_pages or relevant_sources."""
    if "expected_pages" in query:
        return set(query["expected_pages"])
    pages: set[int] = set()
    for src in query.get("relevant_sources", []):
        pages.update(src.get("pages", []))
    return pages


def get_expected_documents(
    query: dict[str, Any], default: list[str] | None = None
) -> set[str]:
    """Extract expected document IDs from query."""
    if "expected_documents" in query:
        return set(query["expected_documents"])
    return set(default or [])
