"""Unit tests for semantic vector search."""

import pytest

from src.retrieval.query_embedding import embed_query
from src.retrieval.vector_search import vector_search
from src.vectorstore.weaviate_client import is_weaviate_ready

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not is_weaviate_ready(),
    reason="Weaviate not running",
)
def test_vector_search_returns_results() -> None:
    """vector_search returns list of results when Weaviate has data."""
    query_vector = embed_query("chave Pix")
    results = vector_search(query_vector, top_k=3)
    assert isinstance(results, list)
    for r in results:
        assert "chunk_id" in r or "text" in r
        assert "document_id" in r
        assert "page_number" in r


@pytest.mark.skipif(
    not is_weaviate_ready(),
    reason="Weaviate not running",
)
def test_vector_search_deterministic() -> None:
    """Same vector produces same results."""
    query_vector = embed_query("cadastro")
    r1 = vector_search(query_vector, top_k=2)
    r2 = vector_search(query_vector, top_k=2)
    assert len(r1) == len(r2)
    if r1:
        assert r1[0].get("chunk_id") == r2[0].get("chunk_id")
