"""Unit tests for query embedding."""

from src.embeddings.validation import BGE_M3_DIMENSIONS
from src.retrieval.query_embedding import embed_query


def test_embed_query_dimensionality() -> None:
    """embed_query returns vector of expected dimension (1024)."""
    vector = embed_query("Como funciona o Pix?")
    assert len(vector) == BGE_M3_DIMENSIONS


def test_embed_query_deterministic() -> None:
    """Same query produces same embedding."""
    v1 = embed_query("Deterministic query.")
    v2 = embed_query("Deterministic query.")
    assert v1 == v2


def test_embed_query_numeric() -> None:
    """Embedding elements are numeric."""
    vector = embed_query("Test query")
    assert all(isinstance(x, (int, float)) for x in vector)
