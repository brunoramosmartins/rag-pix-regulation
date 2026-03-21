"""Evaluation module - Retrieval quality, grounding, answer quality, and hallucination metrics."""

from .answer_quality import (
    AnswerQualityResult,
    compute_answer_quality,
    compute_answer_similarity,
    compute_concept_coverage,
)
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
    average_precision_by_pages,
    evaluate_retrieval,
    evaluate_retrieval_by_pages,
    ndcg_at_k_by_pages,
    ndcg_at_k_graded,
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
    "ndcg_at_k_by_pages",
    "ndcg_at_k_graded",
    "average_precision_by_pages",
    "evaluate_retrieval",
    "evaluate_retrieval_by_pages",
    "RAGEvaluationResult",
    "compute_citation_coverage",
    "detect_hallucination",
    "evaluate_rag_response",
    "AnswerQualityResult",
    "compute_answer_quality",
    "compute_answer_similarity",
    "compute_concept_coverage",
    "run_retrieval_evaluation",
    "run_full_evaluation",
]
