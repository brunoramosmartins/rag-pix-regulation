"""Unit tests for retriever."""

import pytest

from src.retrieval import RetrievalResult, retrieve
from src.vectorstore.weaviate_client import is_weaviate_ready


def test_retrieval_result_has_required_fields() -> None:
    """RetrievalResult dataclass has expected fields."""
    r = RetrievalResult(
        text="Sample",
        chunk_id="id",
        document_id="doc",
        page_number=1,
        section_title="Section",
        similarity_score=0.9,
    )
    assert r.text == "Sample"
    assert r.chunk_id == "id"
    assert r.document_id == "doc"
    assert r.page_number == 1
    assert r.section_title == "Section"
    assert r.similarity_score == 0.9


@pytest.mark.skipif(
    not is_weaviate_ready(),
    reason="Weaviate not running",
)
def test_retrieve_returns_results() -> None:
    """retrieve returns list of RetrievalResult when Weaviate has data."""
    results = retrieve("chave Pix", top_k=3)
    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, RetrievalResult)
        assert r.text
        assert r.chunk_id


@pytest.mark.skipif(
    not is_weaviate_ready(),
    reason="Weaviate not running",
)
def test_retrieve_respects_top_k() -> None:
    """retrieve returns at most top_k results."""
    results = retrieve("cadastro", top_k=2)
    assert len(results) <= 2


@pytest.mark.skipif(
    not is_weaviate_ready(),
    reason="Weaviate not running",
)
def test_retrieve_deterministic() -> None:
    """Same query produces same results."""
    r1 = retrieve("cadastro", top_k=2)
    r2 = retrieve("cadastro", top_k=2)
    assert len(r1) == len(r2)
    if r1:
        assert r1[0].chunk_id == r2[0].chunk_id
