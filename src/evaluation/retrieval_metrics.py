"""Retrieval evaluation metrics: Precision@K and Recall@K (page-based relevance)."""

from pathlib import Path
from typing import Any, Callable

from .dataset_loader import get_expected_pages, load_evaluation_dataset


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


def precision_at_k_by_pages(
    retrieved_pages: list[int],
    expected_pages: set[int],
    k: int,
) -> float:
    """
    Precision@K using page-based relevance.

    Chunk is relevant if chunk.page_number in expected_pages.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_pages[:k]
    hits = sum(1 for p in top_k if p in expected_pages)
    return hits / k


def recall_at_k_by_pages(
    retrieved_pages: list[int],
    expected_pages: set[int],
    k: int,
) -> float:
    """
    Recall@K using page-based relevance.
    """
    if not expected_pages:
        return 0.0
    top_k = retrieved_pages[:k]
    hits = sum(1 for p in top_k if p in expected_pages)
    return hits / len(expected_pages)


def evaluate_retrieval_by_pages(
    dataset_path: Path,
    retriever_fn: Callable[[str], list],
    k: int = 5,
) -> dict[str, float]:
    """
    Evaluate retriever against unified dataset using page-based relevance.

    retriever_fn(query) returns list of objects with .page_number and .document_id.
    """
    data = load_evaluation_dataset(dataset_path)
    precisions: list[float] = []
    recalls: list[float] = []

    for q in data["queries"]:
        expected_pages = get_expected_pages(q)
        if not expected_pages:
            continue

        results = retriever_fn(q["query"])
        retrieved_pages = [
            r.page_number if hasattr(r, "page_number") else r.get("page_number", 0)
            for r in results
        ]

        precisions.append(precision_at_k_by_pages(retrieved_pages, expected_pages, k))
        recalls.append(recall_at_k_by_pages(retrieved_pages, expected_pages, k))

    n = len(precisions)
    if n == 0:
        return {f"precision@{k}": 0.0, f"recall@{k}": 0.0, "n_queries": 0}

    return {
        f"precision@{k}": round(sum(precisions) / n, 4),
        f"recall@{k}": round(sum(recalls) / n, 4),
        "n_queries": n,
    }


def evaluate_retrieval(
    dataset_path: Path,
    retriever_fn: Callable[[str], list],
    k: int = 5,
) -> dict[str, float]:
    """
    Evaluate retriever. Uses page-based relevance for unified dataset.
    """
    return evaluate_retrieval_by_pages(dataset_path, retriever_fn, k)
