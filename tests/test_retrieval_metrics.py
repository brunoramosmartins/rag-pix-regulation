"""Unit tests for retrieval evaluation metrics."""

import tempfile
from pathlib import Path

import pytest

from src.evaluation.retrieval_metrics import (
    evaluate_retrieval,
    precision_at_k,
    precision_at_k_by_pages,
    recall_at_k,
    recall_at_k_by_pages,
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


def test_precision_at_k_by_pages() -> None:
    """Page-based precision: 2 of top-3 pages are expected."""
    retrieved_pages = [1, 2, 5]
    expected_pages = {1, 2}
    assert precision_at_k_by_pages(retrieved_pages, expected_pages, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_by_pages() -> None:
    """Page-based recall: 2 of 3 expected pages found."""
    retrieved_pages = [1, 2, 5]
    expected_pages = {1, 2, 3}
    assert recall_at_k_by_pages(retrieved_pages, expected_pages, k=3) == pytest.approx(2 / 3)


def test_evaluate_retrieval_empty_dataset() -> None:
    """evaluate_retrieval returns zeros when no queries have expected_pages."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('{"queries": [{"query_id": "q1", "query": "x"}]}')
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


def test_load_evaluation_dataset_invalid() -> None:
    """load_evaluation_dataset raises when 'queries' is missing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('{"invalid": "structure"}')
        path = Path(f.name)

    try:
        from src.evaluation.dataset_loader import load_evaluation_dataset

        with pytest.raises(ValueError, match="missing 'queries'"):
            load_evaluation_dataset(path)
    finally:
        path.unlink()


def test_evaluate_retrieval_with_ground_truth() -> None:
    """evaluate_retrieval computes metrics using expected_pages (page-based)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('''{
            "queries": [
                {"query_id": "q1", "query": "x", "expected_pages": [1, 2]}
            ]
        }''')
        path = Path(f.name)

    try:
        class MockResult:
            def __init__(self, page: int):
                self.page_number = page

        # Returns pages [1, 5] - page 1 in expected, page 2 not in top-2
        def mock_retriever(_):
            return [MockResult(1), MockResult(5)]

        metrics = evaluate_retrieval(path, mock_retriever, k=2)
        assert metrics["n_queries"] == 1
        # Precision@2: 1/2 (only page 1 is in expected)
        assert metrics["precision@2"] == 0.5
        # Recall@2: 1/2 (found 1 of 2 expected pages)
        assert metrics["recall@2"] == 0.5
    finally:
        path.unlink()
