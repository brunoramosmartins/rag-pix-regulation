"""Unit tests for evaluation dataset loader."""

import tempfile
from pathlib import Path

import pytest

from src.evaluation.dataset_loader import (
    get_expected_documents,
    get_expected_pages,
    load_evaluation_dataset,
)


def test_load_evaluation_dataset() -> None:
    """Load valid dataset."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('''{
            "queries": [
                {"query_id": "q1", "query": "test?", "expected_pages": [1, 2]}
            ]
        }''')
        path = Path(f.name)

    try:
        data = load_evaluation_dataset(path)
        assert "queries" in data
        assert len(data["queries"]) == 1
        assert data["queries"][0]["query"] == "test?"
    finally:
        path.unlink()


def test_get_expected_pages_from_field() -> None:
    """get_expected_pages uses expected_pages when present."""
    q = {"expected_pages": [1, 2, 3]}
    assert get_expected_pages(q) == {1, 2, 3}


def test_get_expected_pages_from_relevant_sources() -> None:
    """get_expected_pages extracts from relevant_sources when expected_pages absent."""
    q = {"relevant_sources": [{"pages": [5, 6]}, {"pages": [7]}]}
    assert get_expected_pages(q) == {5, 6, 7}


def test_get_expected_documents() -> None:
    """get_expected_documents returns set of document IDs."""
    q = {"expected_documents": ["doc1", "doc2"]}
    assert get_expected_documents(q) == {"doc1", "doc2"}


def test_get_expected_documents_default() -> None:
    """get_expected_documents uses default when absent."""
    q = {}
    assert get_expected_documents(q, default=["manual_dict"]) == {"manual_dict"}
