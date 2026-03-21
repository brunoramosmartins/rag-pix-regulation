"""Unit tests for BM25 keyword search."""

from unittest.mock import MagicMock, patch

import src.retrieval.keyword_search as ks_module


def _make_bm25_obj(chunk_id: str, score: float | None) -> MagicMock:
    """Create a mock Weaviate BM25 response object."""
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


def _mock_keyword_search(objects: list[MagicMock], **kwargs):
    """Run keyword_search with mocked Weaviate client."""
    mock_response = MagicMock()
    mock_response.objects = objects

    mock_collection = MagicMock()
    mock_collection.query.bm25.return_value = mock_response

    mock_client = MagicMock()
    mock_client.collections.get.return_value = mock_collection

    with patch.object(ks_module, "get_weaviate_client", return_value=mock_client):
        results = ks_module.keyword_search(**kwargs)
        return results, mock_collection


def test_keyword_search_returns_results() -> None:
    """BM25 search returns results with correct structure."""
    objects = [
        _make_bm25_obj("c1", score=5.2),
        _make_bm25_obj("c2", score=3.1),
    ]
    results, _ = _mock_keyword_search(objects, query="Art. 3", top_k=5)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["similarity_score"] == 5.2
    assert results[0]["text"] == "text for c1"
    assert results[0]["document_id"] == "doc"
    assert results[0]["page_number"] == 1
    assert results[0]["source_file"] == "test.pdf"


def test_keyword_search_respects_top_k() -> None:
    """Limit parameter is passed to Weaviate BM25 query."""
    objects = [_make_bm25_obj("c1", score=5.0)]
    _, mock_collection = _mock_keyword_search(objects, query="test", top_k=3)
    mock_collection.query.bm25.assert_called_once()
    call_kwargs = mock_collection.query.bm25.call_args
    assert call_kwargs.kwargs["limit"] == 3


def test_keyword_search_default_properties() -> None:
    """Default query_properties is ['text']."""
    objects = [_make_bm25_obj("c1", score=5.0)]
    _, mock_collection = _mock_keyword_search(objects, query="test")
    call_kwargs = mock_collection.query.bm25.call_args
    assert call_kwargs.kwargs["query_properties"] == ["text"]


def test_keyword_search_custom_properties() -> None:
    """Custom query_properties are forwarded to Weaviate."""
    objects = [_make_bm25_obj("c1", score=5.0)]
    custom_props = ["text", "section_title"]
    _, mock_collection = _mock_keyword_search(
        objects, query="test", query_properties=custom_props,
    )
    call_kwargs = mock_collection.query.bm25.call_args
    assert call_kwargs.kwargs["query_properties"] == custom_props


def test_keyword_search_empty_results() -> None:
    """Empty response returns empty list."""
    results, _ = _mock_keyword_search([], query="nonexistent")
    assert results == []
