"""Unit tests for minimum similarity threshold filtering."""

from unittest.mock import MagicMock, patch


def _make_weaviate_obj(chunk_id: str, distance: float | None) -> MagicMock:
    """Create a mock Weaviate response object with given distance."""
    obj = MagicMock()
    obj.properties = {
        "chunk_id": chunk_id,
        "document_id": "doc",
        "page_number": 1,
        "section_title": None,
        "text": f"text for {chunk_id}",
        "source_file": "test.pdf",
    }
    obj.metadata.distance = distance
    return obj


def _mock_vector_search(objects: list[MagicMock], **kwargs):
    """Run vector_search with mocked Weaviate client."""
    mock_response = MagicMock()
    mock_response.objects = objects

    mock_collection = MagicMock()
    mock_collection.query.near_vector.return_value = mock_response

    mock_client = MagicMock()
    mock_client.collections.get.return_value = mock_collection

    dummy_vector = [0.0] * 1024

    with patch("src.retrieval.vector_search.get_weaviate_client", return_value=mock_client):
        from src.retrieval.vector_search import vector_search

        return vector_search(dummy_vector, **kwargs)


def test_min_similarity_filters_low_scores() -> None:
    """Only chunks above min_similarity are returned."""
    objects = [
        _make_weaviate_obj("c1", distance=0.1),  # similarity = 0.9
        _make_weaviate_obj("c2", distance=0.5),  # similarity = 0.5
        _make_weaviate_obj("c3", distance=0.8),  # similarity = 0.2
    ]
    results = _mock_vector_search(objects, top_k=5, min_similarity=0.4)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[1]["chunk_id"] == "c2"


def test_min_similarity_zero_passes_all() -> None:
    """With min_similarity=0.0, all results pass through."""
    objects = [
        _make_weaviate_obj("c1", distance=0.1),
        _make_weaviate_obj("c2", distance=0.9),
    ]
    results = _mock_vector_search(objects, top_k=5, min_similarity=0.0)
    assert len(results) == 2


def test_min_similarity_returns_empty_when_all_below() -> None:
    """When all scores are below threshold, return empty list."""
    objects = [
        _make_weaviate_obj("c1", distance=0.8),  # similarity = 0.2
        _make_weaviate_obj("c2", distance=0.9),  # similarity = 0.1
    ]
    results = _mock_vector_search(objects, top_k=5, min_similarity=0.5)
    assert len(results) == 0


def test_min_similarity_excludes_none_scores() -> None:
    """Chunks with similarity_score=None are excluded when threshold > 0."""
    obj = MagicMock()
    obj.properties = {
        "chunk_id": "c1",
        "document_id": "doc",
        "page_number": 1,
        "section_title": None,
        "text": "text",
        "source_file": "test.pdf",
    }
    obj.metadata.distance = None  # no distance available

    results = _mock_vector_search([obj], top_k=5, min_similarity=0.3)
    assert len(results) == 0


def test_min_similarity_default_preserves_backward_compatibility() -> None:
    """Default min_similarity=0.0 preserves original behavior."""
    objects = [
        _make_weaviate_obj("c1", distance=0.99),  # similarity = 0.01
    ]
    results = _mock_vector_search(objects, top_k=5)
    assert len(results) == 1
