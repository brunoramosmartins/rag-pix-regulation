"""Evaluation module - Retrieval quality, grounding, and hallucination metrics."""

from .retrieval_dataset import load_retrieval_dataset
from .retrieval_metrics import (
    evaluate_queries,
    evaluate_retrieval,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "load_retrieval_dataset",
    "precision_at_k",
    "recall_at_k",
    "evaluate_queries",
    "evaluate_retrieval",
]
