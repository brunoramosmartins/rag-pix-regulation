"""Unit tests for retrieval evaluation metrics."""

import tempfile
from pathlib import Path

import pytest

from src.evaluation.retrieval_metrics import (
    average_precision_by_pages,
    evaluate_retrieval,
    ndcg_at_k_by_pages,
    ndcg_at_k_graded,
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
    assert precision_at_k_by_pages(
        retrieved_pages, expected_pages, k=3
    ) == pytest.approx(2 / 3)


def test_recall_at_k_by_pages() -> None:
    """Page-based recall: 2 of 3 expected pages found."""
    retrieved_pages = [1, 2, 5]
    expected_pages = {1, 2, 3}
    assert recall_at_k_by_pages(retrieved_pages, expected_pages, k=3) == pytest.approx(
        2 / 3
    )


def test_recall_at_k_by_pages_deduplicated() -> None:
    """Recall@K must not exceed 1.0 when multiple chunks come from the same page."""
    retrieved_pages = [122, 122, 122, 122, 122]
    expected_pages = {122}
    assert recall_at_k_by_pages(retrieved_pages, expected_pages, k=5) == 1.0


def test_recall_at_k_by_pages_dedup_multiple_expected() -> None:
    """Recall deduplication: 3 chunks from page 1, 2 from page 2, expected {1,2,3}."""
    retrieved_pages = [1, 1, 1, 2, 2]
    expected_pages = {1, 2, 3}
    # unique hits: {1, 2} ∩ {1, 2, 3} = {1, 2} -> 2/3
    assert recall_at_k_by_pages(retrieved_pages, expected_pages, k=5) == pytest.approx(
        2 / 3
    )


def test_recall_at_k_by_pages_empty_expected() -> None:
    """Recall@K = 0 when expected_pages is empty."""
    assert recall_at_k_by_pages([1, 2, 3], set(), k=3) == 0.0


# --- NDCG@K tests ---


def test_ndcg_at_k_perfect_ranking() -> None:
    """NDCG@K = 1.0 when all relevant results are at the top."""
    retrieved_pages = [1, 2, 5, 6, 7]
    expected_pages = {1, 2}
    assert ndcg_at_k_by_pages(retrieved_pages, expected_pages, k=5) == pytest.approx(
        1.0
    )


def test_ndcg_at_k_worst_ranking() -> None:
    """NDCG@K < 1.0 when relevant results are at the bottom."""
    retrieved_pages = [5, 6, 7, 1, 2]
    expected_pages = {1, 2}
    ndcg = ndcg_at_k_by_pages(retrieved_pages, expected_pages, k=5)
    assert ndcg < 1.0
    assert ndcg > 0.0


def test_ndcg_at_k_no_relevant() -> None:
    """NDCG@K = 0 when no relevant results."""
    retrieved_pages = [5, 6, 7]
    expected_pages = {1, 2}
    assert ndcg_at_k_by_pages(retrieved_pages, expected_pages, k=3) == 0.0


def test_ndcg_at_k_empty_expected() -> None:
    """NDCG@K = 0 when expected_pages is empty."""
    assert ndcg_at_k_by_pages([1, 2, 3], set(), k=3) == 0.0


def test_ndcg_at_k_zero_k() -> None:
    """NDCG@K = 0 when k=0."""
    assert ndcg_at_k_by_pages([1, 2], {1}, k=0) == 0.0


def test_ndcg_at_k_single_relevant_at_top() -> None:
    """NDCG@K = 1.0 when the only relevant result is at position 1."""
    retrieved_pages = [1, 5, 6]
    expected_pages = {1}
    assert ndcg_at_k_by_pages(retrieved_pages, expected_pages, k=3) == pytest.approx(
        1.0
    )


def test_ndcg_at_k_single_relevant_at_bottom() -> None:
    """NDCG@K < 1.0 when the only relevant result is at position 3."""
    retrieved_pages = [5, 6, 1]
    expected_pages = {1}
    ndcg = ndcg_at_k_by_pages(retrieved_pages, expected_pages, k=3)
    assert ndcg < 1.0
    assert ndcg > 0.0


# --- Average Precision (MAP component) tests ---


def test_ap_perfect_ranking() -> None:
    """AP = 1.0 when all relevant at top."""
    retrieved_pages = [1, 2, 5, 6]
    expected_pages = {1, 2}
    assert average_precision_by_pages(
        retrieved_pages, expected_pages, k=4
    ) == pytest.approx(1.0)


def test_ap_worst_ranking() -> None:
    """AP < 1.0 when relevant results are at the bottom."""
    retrieved_pages = [5, 6, 1, 2]
    expected_pages = {1, 2}
    ap = average_precision_by_pages(retrieved_pages, expected_pages, k=4)
    # Position 3: prec=1/3, position 4: prec=2/4=0.5
    # AP = (1/2) * (1/3 + 0.5) = (1/2) * (5/6) = 5/12
    assert ap == pytest.approx(5 / 12)


def test_ap_no_relevant() -> None:
    """AP = 0 when no relevant results found."""
    retrieved_pages = [5, 6, 7]
    expected_pages = {1, 2}
    assert average_precision_by_pages(retrieved_pages, expected_pages, k=3) == 0.0


def test_ap_empty_expected() -> None:
    """AP = 0 when expected_pages is empty."""
    assert average_precision_by_pages([1, 2], set(), k=2) == 0.0


def test_ap_zero_k() -> None:
    """AP = 0 when k=0."""
    assert average_precision_by_pages([1, 2], {1}, k=0) == 0.0


def test_ap_interleaved() -> None:
    """AP with interleaved relevant/irrelevant results."""
    retrieved_pages = [1, 5, 2, 6]
    expected_pages = {1, 2}
    # Position 1: prec=1/1, position 3: prec=2/3
    # AP = (1/2) * (1 + 2/3) = (1/2) * (5/3) = 5/6
    ap = average_precision_by_pages(retrieved_pages, expected_pages, k=4)
    assert ap == pytest.approx(5 / 6)


# --- Integration tests ---


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
        assert metrics["ndcg@5"] == 0.0
        assert metrics["map@5"] == 0.0
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
    """evaluate_retrieval computes all metrics using expected_pages."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("""{
            "queries": [
                {"query_id": "q1", "query": "x", "expected_pages": [1, 2]}
            ]
        }""")
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
        # Recall@2: 1/2 (found 1 of 2 expected pages, deduplicated)
        assert metrics["recall@2"] == 0.5
        # NDCG and MAP should be present
        assert "ndcg@2" in metrics
        assert "map@2" in metrics
    finally:
        path.unlink()


def test_evaluate_retrieval_with_duplicate_pages() -> None:
    """evaluate_retrieval: recall is correctly deduplicated when chunks share pages."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("""{
            "queries": [
                {"query_id": "q1", "query": "x", "expected_pages": [122]}
            ]
        }""")
        path = Path(f.name)

    try:

        class MockResult:
            def __init__(self, page: int):
                self.page_number = page

        # 3 chunks from the same page — recall must be 1.0, not 3.0
        def mock_retriever(_):
            return [MockResult(122), MockResult(122), MockResult(122)]

        metrics = evaluate_retrieval(path, mock_retriever, k=3)
        assert metrics["recall@3"] == 1.0
        assert metrics["recall@3"] <= 1.0  # never exceeds 1.0
    finally:
        path.unlink()


# --- Graded NDCG tests ---


def test_ndcg_graded_perfect() -> None:
    """Graded NDCG = 1.0 when highest-relevance chunks are at the top."""
    retrieved = ["c1", "c2", "c3"]
    annotations = [
        {"chunk_id": "c1", "relevance": 2},
        {"chunk_id": "c2", "relevance": 1},
        {"chunk_id": "c3", "relevance": 0},
    ]
    assert ndcg_at_k_graded(retrieved, annotations, k=3) == pytest.approx(1.0)


def test_ndcg_graded_reversed() -> None:
    """Graded NDCG < 1.0 when irrelevant chunks are at top."""
    retrieved = ["c3", "c2", "c1"]
    annotations = [
        {"chunk_id": "c1", "relevance": 2},
        {"chunk_id": "c2", "relevance": 1},
        {"chunk_id": "c3", "relevance": 0},
    ]
    ndcg = ndcg_at_k_graded(retrieved, annotations, k=3)
    assert ndcg < 1.0
    assert ndcg > 0.0


def test_ndcg_graded_no_annotations() -> None:
    """Graded NDCG = 0 when no annotations provided."""
    assert ndcg_at_k_graded(["c1", "c2"], [], k=2) == 0.0


def test_ndcg_graded_unknown_chunks() -> None:
    """Graded NDCG handles chunks not in annotations (treated as relevance=0)."""
    retrieved = ["unknown1", "unknown2", "c1"]
    annotations = [
        {"chunk_id": "c1", "relevance": 2},
    ]
    ndcg = ndcg_at_k_graded(retrieved, annotations, k=3)
    assert ndcg < 1.0
    assert ndcg > 0.0


def test_ndcg_graded_all_irrelevant() -> None:
    """Graded NDCG = 0 when all annotations have relevance 0."""
    retrieved = ["c1", "c2"]
    annotations = [
        {"chunk_id": "c1", "relevance": 0},
        {"chunk_id": "c2", "relevance": 0},
    ]
    assert ndcg_at_k_graded(retrieved, annotations, k=2) == 0.0
