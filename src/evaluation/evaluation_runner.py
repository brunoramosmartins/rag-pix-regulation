"""Evaluation pipeline orchestration — retrieval and RAG evaluation."""

import json
import logging
from pathlib import Path
from typing import Any, Callable

from .dataset_loader import get_expected_pages, load_evaluation_dataset
from .rag_evaluation import RAGEvaluationResult, evaluate_rag_response
from .retrieval_metrics import (
    evaluate_retrieval_by_pages,
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
        if not expected_pages:
            continue

        try:
            response = rag_fn(q["query"])
        except Exception as e:
            logger.warning("RAG failed for query %s: %s", q.get("query_id"), e)
            continue

        answer = response.answer if hasattr(response, "answer") else response.get("answer", "")
        context = response.context if hasattr(response, "context") else response.get("context", "")
        citations = response.citations if hasattr(response, "citations") else response.get("citations", [])
        chunks = response.retrieved_chunks if hasattr(response, "retrieved_chunks") else response.get("retrieved_chunks", [])

        retrieved_pages = [c.page_number if hasattr(c, "page_number") else c.get("page_number", 0) for c in chunks]
        prec = precision_at_k_by_pages(retrieved_pages, expected_pages, k)
        rec = recall_at_k_by_pages(retrieved_pages, expected_pages, k)

        result = evaluate_rag_response(
            query_id=q.get("query_id", ""),
            answer=answer,
            context=context,
            citations=citations,
            retrieved_chunks=chunks,
            expected_pages=expected_pages,
            precision_at_k=prec,
            recall_at_k=rec,
        )

        per_query.append({
            "query_id": result.query_id,
            "precision_at_k": result.precision_at_k,
            "recall_at_k": result.recall_at_k,
            "citation_coverage": result.citation_coverage,
            "groundedness_score": result.groundedness_score,
            "hallucination_detected": result.hallucination_detected,
        })
        citation_coverages.append(result.citation_coverage)
        groundedness_scores.append(result.groundedness_score)
        if result.hallucination_detected:
            hallucination_count += 1

    n_rag = len(per_query)
    return {
        "retrieval": retrieval_metrics,
        "rag": {
            "citation_coverage": round(sum(citation_coverages) / n_rag, 4) if n_rag else 0.0,
            "groundedness_avg": round(sum(groundedness_scores) / n_rag, 4) if n_rag else 0.0,
            "hallucination_rate": round(hallucination_count / n_rag, 4) if n_rag else 0.0,
            "n_queries": n_rag,
        },
        "per_query": per_query,
    }


def export_report(results: dict[str, Any], output_path: Path) -> None:
    """Export evaluation results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", output_path)
