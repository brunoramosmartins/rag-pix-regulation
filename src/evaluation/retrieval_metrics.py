"""Retrieval evaluation metrics: Precision@K, Recall@K, NDCG@K, MAP@K (page-based relevance)."""

import math
from pathlib import Path
from typing import Callable

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
    Recall@K using page-based relevance (deduplicated).

    Counts unique expected pages found in top-K, preventing recall > 1.0
    when multiple chunks come from the same page.
    """
    if not expected_pages:
        return 0.0
    top_k = retrieved_pages[:k]
    unique_hits = len(set(top_k) & expected_pages)
    return unique_hits / len(expected_pages)


def ndcg_at_k_by_pages(
    retrieved_pages: list[int],
    expected_pages: set[int],
    k: int,
) -> float:
    """
    NDCG@K using page-based binary relevance.

    Measures ranking quality: rewards relevant results appearing at higher positions.
    Uses binary relevance (1 if page in expected_pages, 0 otherwise).
    """
    if k <= 0 or not expected_pages:
        return 0.0

    top_k = retrieved_pages[:k]

    # DCG: sum of relevance / log2(rank + 1)
    dcg = 0.0
    for i, page in enumerate(top_k):
        rel = 1.0 if page in expected_pages else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because rank starts at 1, log2(1+1)

    # Ideal DCG: all relevant results at the top
    n_relevant = min(len(expected_pages), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def average_precision_by_pages(
    retrieved_pages: list[int],
    expected_pages: set[int],
    k: int,
) -> float:
    """
    Average Precision@K using page-based relevance.

    AP = (1/|relevant|) * sum(Precision@i * rel(i)) for i=1..K.
    Measures both precision and ranking quality in a single metric.
    """
    if k <= 0 or not expected_pages:
        return 0.0

    top_k = retrieved_pages[:k]
    hits = 0
    sum_precision = 0.0

    for i, page in enumerate(top_k):
        if page in expected_pages:
            hits += 1
            sum_precision += hits / (i + 1)

    if hits == 0:
        return 0.0

    return sum_precision / len(expected_pages)


def ndcg_at_k_graded(
    retrieved_chunk_ids: list[str],
    chunk_annotations: list[dict],
    k: int,
) -> float:
    """
    NDCG@K using graded relevance from chunk annotations.

    chunk_annotations: list of {"chunk_id": str, "relevance": int (0-2)}.
    Uses graded relevance instead of binary, providing finer-grained ranking quality.
    Falls back gracefully if annotations are missing.
    """
    if k <= 0 or not chunk_annotations:
        return 0.0

    # Build relevance lookup
    rel_map = {a["chunk_id"]: a["relevance"] for a in chunk_annotations}

    top_k = retrieved_chunk_ids[:k]

    # DCG with graded relevance
    dcg = 0.0
    for i, cid in enumerate(top_k):
        rel = float(rel_map.get(cid, 0))
        dcg += rel / math.log2(i + 2)

    # Ideal DCG: sort all relevance scores descending
    all_rels = sorted([a["relevance"] for a in chunk_annotations], reverse=True)
    ideal_rels = all_rels[:k]
    idcg = sum(float(r) / math.log2(i + 2) for i, r in enumerate(ideal_rels))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def evaluate_retrieval_by_pages(
    dataset_path: Path,
    retriever_fn: Callable[[str], list],
    k: int = 5,
) -> dict[str, float]:
    """
    Evaluate retriever against unified dataset using page-based relevance.

    retriever_fn(query) returns list of objects with .page_number and .document_id.
    Returns Precision@K, Recall@K, NDCG@K, and MAP@K.
    """
    data = load_evaluation_dataset(dataset_path)
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    avg_precisions: list[float] = []

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
        ndcgs.append(ndcg_at_k_by_pages(retrieved_pages, expected_pages, k))
        avg_precisions.append(
            average_precision_by_pages(retrieved_pages, expected_pages, k)
        )

    n = len(precisions)
    if n == 0:
        return {
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            f"map@{k}": 0.0,
            "n_queries": 0,
        }

    return {
        f"precision@{k}": round(sum(precisions) / n, 4),
        f"recall@{k}": round(sum(recalls) / n, 4),
        f"ndcg@{k}": round(sum(ndcgs) / n, 4),
        f"map@{k}": round(sum(avg_precisions) / n, 4),
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
