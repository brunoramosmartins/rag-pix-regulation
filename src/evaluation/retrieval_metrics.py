"""Retrieval evaluation metrics: Precision@K and Recall@K."""

from pathlib import Path
from typing import Any, Callable

from .retrieval_dataset import load_retrieval_dataset


def evaluate_queries(
    queries: list[dict[str, Any]],
    retriever_fn: Callable[[str], list],
    k: int = 5,
) -> tuple[list[float], list[float]]:
    """
    Evaluate retriever on a list of query dicts.

    Each query must have "query" and "relevant_chunks" keys.
    Skips queries with empty relevant_chunks.
    Returns (precisions, recalls) per query.
    """
    precisions: list[float] = []
    recalls: list[float] = []

    for q in queries:
        relevant = set(q.get("relevant_chunks", []))
        if not relevant:
            continue

        results = retriever_fn(q["query"])
        retrieved_ids = [
            r.chunk_id if hasattr(r, "chunk_id") else r.get("chunk_id", "")
            for r in results
        ]

        precisions.append(precision_at_k(retrieved_ids, relevant, k))
        recalls.append(recall_at_k(retrieved_ids, relevant, k))

    return precisions, recalls


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Precision@K = |retrieved ∩ relevant| / K.

    Measures fraction of top-K results that are relevant.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Recall@K = |retrieved ∩ relevant| / |relevant|.

    Measures fraction of relevant chunks found in top-K.
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(relevant_ids)


def evaluate_retrieval(
    dataset_path: Path,
    retriever_fn: Callable[[str], list],
    k: int = 5,
) -> dict[str, float]:
    """
    Evaluate retriever against dataset file.

    Loads dataset, calls evaluate_queries, returns aggregated metrics.
    """
    data = load_retrieval_dataset(dataset_path)
    precisions, recalls = evaluate_queries(data["queries"], retriever_fn, k)

    n = len(precisions)
    if n == 0:
        return {f"precision@{k}": 0.0, f"recall@{k}": 0.0, "n_queries": 0}

    return {
        f"precision@{k}": round(sum(precisions) / n, 4),
        f"recall@{k}": round(sum(recalls) / n, 4),
        "n_queries": n,
    }
