"""Evaluation pipeline orchestration — retrieval and RAG evaluation."""

import json
import logging
from pathlib import Path
from typing import Any, Callable

from .dataset_loader import get_expected_pages, load_evaluation_dataset
from .rag_evaluation import evaluate_rag_response
from .retrieval_metrics import (
    average_precision_by_pages,
    evaluate_retrieval_by_pages,
    ndcg_at_k_by_pages,
    precision_at_k_by_pages,
    recall_at_k_by_pages,
)

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path("data/evaluation/rag_evaluation_dataset.json")
DEFAULT_REPORTS_DIR = Path("reports")


def run_retrieval_evaluation(
    dataset_path: Path,
    retriever_fn: Callable[[str], list],
    k: int = 5,
) -> dict[str, float]:
    """
    Run retrieval evaluation only.

    Returns aggregated Precision@K and Recall@K.
    """
    return evaluate_retrieval_by_pages(dataset_path, retriever_fn, k)


def run_full_evaluation(
    dataset_path: Path,
    retriever_fn: Callable[[str], list],
    rag_fn: Callable[[str], Any] | None,
    k: int = 5,
) -> dict[str, Any]:
    """
    Run full RAG evaluation: retrieval metrics + groundedness + hallucination.

    rag_fn(query) returns object with: answer, context, citations, retrieved_chunks.
    If rag_fn is None, runs retrieval-only evaluation.
    """
    data = load_evaluation_dataset(dataset_path)
    retrieval_metrics = run_retrieval_evaluation(dataset_path, retriever_fn, k)

    if rag_fn is None:
        return {
            "retrieval": retrieval_metrics,
            "rag": None,
            "per_query": [],
        }

    per_query: list[dict[str, Any]] = []
    citation_coverages: list[float] = []
    groundedness_scores: list[float] = []
    hallucination_count = 0

    for q in data["queries"]:
        expected_pages = get_expected_pages(q)
        difficulty = q.get("difficulty", "unknown")

        # Skip negative queries (no expected pages) for retrieval metrics
        if not expected_pages:
            continue

        try:
            response = rag_fn(q["query"])
        except Exception as e:
            logger.warning("RAG failed for query %s: %s", q.get("query_id"), e)
            continue

        answer = (
            response.answer
            if hasattr(response, "answer")
            else response.get("answer", "")
        )
        context = (
            response.context
            if hasattr(response, "context")
            else response.get("context", "")
        )
        citations = (
            response.citations
            if hasattr(response, "citations")
            else response.get("citations", [])
        )
        chunks = (
            response.retrieved_chunks
            if hasattr(response, "retrieved_chunks")
            else response.get("retrieved_chunks", [])
        )

        retrieved_pages = [
            c.page_number if hasattr(c, "page_number") else c.get("page_number", 0)
            for c in chunks
        ]
        prec = precision_at_k_by_pages(retrieved_pages, expected_pages, k)
        rec = recall_at_k_by_pages(retrieved_pages, expected_pages, k)
        ndcg = ndcg_at_k_by_pages(retrieved_pages, expected_pages, k)
        ap = average_precision_by_pages(retrieved_pages, expected_pages, k)

        result = evaluate_rag_response(
            query_id=q.get("query_id", ""),
            answer=answer,
            context=context,
            citations=citations,
            retrieved_chunks=chunks,
            expected_pages=expected_pages,
            precision_at_k=prec,
            recall_at_k=rec,
            expected_answer_summary=q.get("expected_answer_summary", ""),
            key_concepts=q.get("key_concepts", []),
        )

        per_query.append(
            {
                "query_id": result.query_id,
                "difficulty": difficulty,
                "precision_at_k": result.precision_at_k,
                "recall_at_k": result.recall_at_k,
                "ndcg_at_k": ndcg,
                "average_precision": ap,
                "citation_coverage": result.citation_coverage,
                "groundedness_score": result.groundedness_score,
                "hallucination_detected": result.hallucination_detected,
                "answer_similarity": result.answer_similarity,
                "concept_coverage": result.concept_coverage,
                "quality_score": result.quality_score,
            }
        )
        citation_coverages.append(result.citation_coverage)
        groundedness_scores.append(result.groundedness_score)
        if result.hallucination_detected:
            hallucination_count += 1

    n_rag = len(per_query)

    # Compute metrics by difficulty tier
    by_difficulty = _aggregate_by_difficulty(per_query)

    # Aggregate answer quality metrics
    answer_sims = [q["answer_similarity"] for q in per_query]
    concept_covs = [q["concept_coverage"] for q in per_query]
    quality_scores = [q["quality_score"] for q in per_query]

    return {
        "retrieval": retrieval_metrics,
        "rag": {
            "citation_coverage": round(sum(citation_coverages) / n_rag, 4)
            if n_rag
            else 0.0,
            "groundedness_avg": round(sum(groundedness_scores) / n_rag, 4)
            if n_rag
            else 0.0,
            "hallucination_rate": round(hallucination_count / n_rag, 4)
            if n_rag
            else 0.0,
            "answer_similarity_avg": round(sum(answer_sims) / n_rag, 4)
            if n_rag
            else 0.0,
            "concept_coverage_avg": round(sum(concept_covs) / n_rag, 4)
            if n_rag
            else 0.0,
            "quality_score_avg": round(sum(quality_scores) / n_rag, 4)
            if n_rag
            else 0.0,
            "n_queries": n_rag,
        },
        "by_difficulty": by_difficulty,
        "per_query": per_query,
    }


def _aggregate_by_difficulty(
    per_query: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Aggregate per-query metrics by difficulty tier."""
    tiers: dict[str, list[dict[str, Any]]] = {}
    for q in per_query:
        tier = q.get("difficulty", "unknown")
        tiers.setdefault(tier, []).append(q)

    result: dict[str, dict[str, float]] = {}
    for tier, queries in sorted(tiers.items()):
        n = len(queries)
        result[tier] = {
            "n_queries": n,
            "precision_at_k": round(sum(q["precision_at_k"] for q in queries) / n, 4),
            "recall_at_k": round(sum(q["recall_at_k"] for q in queries) / n, 4),
            "ndcg_at_k": round(sum(q["ndcg_at_k"] for q in queries) / n, 4),
            "average_precision": round(
                sum(q["average_precision"] for q in queries) / n, 4
            ),
            "answer_similarity": round(
                sum(q.get("answer_similarity", 0) for q in queries) / n, 4
            ),
            "concept_coverage": round(
                sum(q.get("concept_coverage", 0) for q in queries) / n, 4
            ),
            "quality_score": round(
                sum(q.get("quality_score", 0) for q in queries) / n, 4
            ),
        }
    return result


def export_report(results: dict[str, Any], output_path: Path) -> None:
    """Export evaluation results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", output_path)
