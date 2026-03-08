"""Unit tests for retrieval evaluation metrics."""

import tempfile
from pathlib import Path

import pytest

from src.evaluation.retrieval_metrics import (
    evaluate_retrieval,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k_perfect() -> None:
    """Precision@K = 1 when all top-K are relevant."""
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b", "c"}
    assert precision_at_k(retrieved, relevant, k=3) == 1.0


def test_precision_at_k_partial() -> None:
    """Precision@K = 2/3 when 2 of top-3 are relevant."""
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3, rel=1e-5)


def test_precision_at_k_zero() -> None:
    """Precision@K = 0 when none of top-K are relevant."""
    retrieved = ["x", "y", "z"]
    relevant = {"a", "b", "c"}
    assert precision_at_k(retrieved, relevant, k=3) == 0.0


def test_recall_at_k_perfect() -> None:
    """Recall@K = 1 when all relevant are in top-K."""
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_recall_at_k_partial() -> None:
    """Recall@K = 0.5 when half of relevant are in top-K."""
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b", "c", "d"}
    assert recall_at_k(retrieved, relevant, k=3) == 0.75


def test_recall_at_k_empty_relevant() -> None:
    """Recall@K = 0 when no relevant chunks."""
    retrieved = ["a", "b"]
    relevant = set()
    assert recall_at_k(retrieved, relevant, k=2) == 0.0


def test_evaluate_retrieval_empty_dataset() -> None:
    """evaluate_retrieval returns zeros when no queries have ground truth."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('{"queries": [{"query_id": "q1", "query": "x", "relevant_chunks": []}]}')
        path = Path(f.name)

    try:
        def mock_retriever(_):
            return []

        metrics = evaluate_retrieval(path, mock_retriever, k=5)
        assert metrics["n_queries"] == 0
        assert metrics["precision@5"] == 0.0
        assert metrics["recall@5"] == 0.0
    finally:
        path.unlink()


def test_load_retrieval_dataset_invalid() -> None:
    """load_retrieval_dataset raises when 'queries' is missing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('{"invalid": "structure"}')
        path = Path(f.name)

    try:
        from src.evaluation.retrieval_dataset import load_retrieval_dataset

        with pytest.raises(ValueError, match="missing 'queries'"):
            load_retrieval_dataset(path)
    finally:
        path.unlink()


def test_evaluate_retrieval_with_ground_truth() -> None:
    """evaluate_retrieval computes metrics for queries with relevant_chunks."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('''{
            "queries": [
                {"query_id": "q1", "query": "x", "relevant_chunks": ["a", "b"]}
            ]
        }''')
        path = Path(f.name)

    try:
        # Mock retriever returns [a, c] - 1 relevant in top-2
        class MockResult:
            def __init__(self, cid):
                self.chunk_id = cid

        def mock_retriever(_):
            return [MockResult("a"), MockResult("c")]

        metrics = evaluate_retrieval(path, mock_retriever, k=2)
        assert metrics["n_queries"] == 1
        # Precision@2: 1/2 relevant in top-2
        assert metrics["precision@2"] == 0.5
        # Recall@2: 1/2 relevant found
        assert metrics["recall@2"] == 0.5
    finally:
        path.unlink()
