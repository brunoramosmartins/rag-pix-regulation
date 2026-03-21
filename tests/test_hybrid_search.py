"""Unit tests for hybrid search (BM25 + vector)."""

import pytest
from unittest.mock import MagicMock, patch

import src.retrieval.hybrid_search as hs_module


def _make_hybrid_obj(chunk_id: str, score: float | None) -> MagicMock:
    """Create a mock Weaviate hybrid response object."""
    obj = MagicMock()
    obj.properties = {
        "chunk_id": chunk_id,
        "document_id": "doc",
        "page_number": 1,
        "section_title": "Section 1",
        "text": f"text for {chunk_id}",
        "source_file": "test.pdf",
    }
    obj.metadata.score = score
    return obj


def _mock_hybrid_search(objects: list[MagicMock], **kwargs):
    """Run hybrid_search with mocked Weaviate client."""
    mock_response = MagicMock()
    mock_response.objects = objects

    mock_collection = MagicMock()
    mock_collection.query.hybrid.return_value = mock_response

    mock_client = MagicMock()
    mock_client.collections.get.return_value = mock_collection

    dummy_vector = [0.0] * 1024

    with patch.object(hs_module, "get_weaviate_client", return_value=mock_client):
        if "query_vector" not in kwargs:
            kwargs["query_vector"] = dummy_vector
        results = hs_module.hybrid_search(**kwargs)
        return results, mock_collection


def test_hybrid_search_returns_results() -> None:
    """Hybrid search returns results with correct structure."""
    objects = [
        _make_hybrid_obj("c1", score=0.85),
        _make_hybrid_obj("c2", score=0.62),
    ]
    results, _ = _mock_hybrid_search(objects, query="chave Pix")
    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["similarity_score"] == 0.85
    assert results[0]["text"] == "text for c1"
    assert results[0]["document_id"] == "doc"


def test_hybrid_search_passes_alpha_and_vector() -> None:
    """Alpha and vector are forwarded to Weaviate hybrid query."""
    objects = [_make_hybrid_obj("c1", score=0.8)]
    dummy_vector = [0.0] * 1024
    _, mock_collection = _mock_hybrid_search(
        objects, query="test", query_vector=dummy_vector, alpha=0.7,
    )
    call_kwargs = mock_collection.query.hybrid.call_args.kwargs
    assert call_kwargs["alpha"] == 0.7
    assert call_kwargs["vector"] == dummy_vector


def test_hybrid_search_default_fusion_type() -> None:
    """Default fusion type is 'ranked' (RRF)."""
    from weaviate.classes.query import HybridFusion

    objects = [_make_hybrid_obj("c1", score=0.8)]
    _, mock_collection = _mock_hybrid_search(objects, query="test")
    call_kwargs = mock_collection.query.hybrid.call_args.kwargs
    assert call_kwargs["fusion_type"] == HybridFusion.RANKED


def test_hybrid_search_relative_score_fusion() -> None:
    """Relative score fusion type is correctly applied."""
    from weaviate.classes.query import HybridFusion

    objects = [_make_hybrid_obj("c1", score=0.8)]
    _, mock_collection = _mock_hybrid_search(
        objects, query="test", fusion_type="relative_score",
    )
    call_kwargs = mock_collection.query.hybrid.call_args.kwargs
    assert call_kwargs["fusion_type"] == HybridFusion.RELATIVE_SCORE


def test_hybrid_search_validates_vector_dimensions() -> None:
    """Invalid embedding dimensions raise ValueError."""
    with pytest.raises(ValueError, match="Invalid embedding dimension"):
        _mock_hybrid_search([], query="test", query_vector=[0.0] * 512)


def test_hybrid_search_validates_alpha_range() -> None:
    """Alpha outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError, match="alpha must be between"):
        _mock_hybrid_search([], query="test", query_vector=[0.0] * 1024, alpha=1.5)


def test_hybrid_search_empty_results() -> None:
    """Empty response returns empty list."""
    results, _ = _mock_hybrid_search([], query="nonexistent")
    assert results == []
