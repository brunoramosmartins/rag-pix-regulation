"""Evaluation module - Retrieval quality, grounding, and hallucination metrics."""

from .dataset_loader import (
    get_expected_documents,
    get_expected_pages,
    load_evaluation_dataset,
)
from .rag_evaluation import (
    RAGEvaluationResult,
    compute_citation_coverage,
    detect_hallucination,
    evaluate_rag_response,
)
from .retrieval_metrics import (
    evaluate_retrieval,
    evaluate_retrieval_by_pages,
    precision_at_k,
    precision_at_k_by_pages,
    recall_at_k,
    recall_at_k_by_pages,
)
from .evaluation_runner import run_full_evaluation, run_retrieval_evaluation

__all__ = [
    "load_evaluation_dataset",
    "get_expected_pages",
    "get_expected_documents",
    "precision_at_k",
    "recall_at_k",
    "precision_at_k_by_pages",
    "recall_at_k_by_pages",
    "evaluate_retrieval",
    "evaluate_retrieval_by_pages",
    "RAGEvaluationResult",
    "compute_citation_coverage",
    "detect_hallucination",
    "evaluate_rag_response",
    "run_retrieval_evaluation",
    "run_full_evaluation",
]
