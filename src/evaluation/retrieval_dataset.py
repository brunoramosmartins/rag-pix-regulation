"""Loader for retrieval evaluation dataset."""

import json
from pathlib import Path
from typing import Any


def load_retrieval_dataset(path: Path) -> dict[str, Any]:
    """
    Load retrieval evaluation dataset from JSON.

    Expected format:
    {
      "queries": [
        {"query_id": "q1", "query": "...", "relevant_chunks": ["chunk_id_1", ...]}
      ]
    }
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "queries" not in data:
        raise ValueError("Invalid retrieval dataset: missing 'queries' field")

    for i, q in enumerate(data["queries"]):
        if "query" not in q:
            raise ValueError(f"Invalid query entry at index {i}: missing 'query' field")

    return data


def get_queries_with_relevant(path: Path) -> list[tuple[str, str, set[str]]]:
    """
    Yield (query_id, query, relevant_chunk_ids) for each query that has ground truth.

    Skips queries with empty relevant_chunks.
    """
    data = load_retrieval_dataset(path)
    for q in data.get("queries", []):
        relevant = set(q.get("relevant_chunks", []))
        if relevant:
            yield q["query_id"], q["query"], relevant
