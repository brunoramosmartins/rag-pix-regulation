"""Unit tests for semantic vector search."""

import pytest

from src.retrieval.vector_search import search
from src.vectorstore.weaviate_client import is_weaviate_ready


@pytest.mark.skipif(
    not is_weaviate_ready(),
    reason="Weaviate not running",
)
def test_search_returns_results() -> None:
    """search returns list of results when Weaviate has data."""
    results = search("chave Pix", top_k=3)
    assert isinstance(results, list)
    for r in results:
        assert "chunk_id" in r or "text" in r
        assert "document_id" in r
        assert "page_number" in r


@pytest.mark.skipif(
    not is_weaviate_ready(),
    reason="Weaviate not running",
)
def test_search_deterministic() -> None:
    """Same query produces same results."""
    r1 = search("cadastro", top_k=2)
    r2 = search("cadastro", top_k=2)
    assert len(r1) == len(r2)
    if r1:
        assert r1[0].get("chunk_id") == r2[0].get("chunk_id")
