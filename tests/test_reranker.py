"""Unit tests for cross-encoder reranker."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch


@dataclass
class _RetrievalResult:
    """Minimal RetrievalResult for tests (avoids heavy imports)."""
    text: str
    chunk_id: str
    document_id: str
    page_number: int
    section_title: str | None
    similarity_score: float | None
    source_file: str | None = None


def _chunk(
    text: str = "chunk text",
    chunk_id: str = "c1",
    score: float = 0.5,
    page: int = 1,
) -> _RetrievalResult:
    return _RetrievalResult(
        text=text,
        chunk_id=chunk_id,
        document_id="doc",
        page_number=page,
        section_title=None,
        similarity_score=score,
    )


def test_rerank_reorders_by_cross_encoder_score() -> None:
    """Cross-encoder scores should determine final ordering."""
    chunks = [_chunk(chunk_id="c1"), _chunk(chunk_id="c2"), _chunk(chunk_id="c3")]
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = [0.1, 0.9, 0.5]  # c2 is best

    with patch("src.retrieval.reranker._get_cross_encoder", return_value=mock_encoder):
        from src.retrieval.reranker import rerank

        result = rerank("query", chunks, top_n=2)

    assert len(result) == 2
    assert result[0].chunk_id == "c2"  # highest score (0.9)
    assert result[1].chunk_id == "c3"  # second highest (0.5)


def test_rerank_updates_similarity_score() -> None:
    """Reranked results should have cross-encoder scores, not original similarity."""
    chunks = [_chunk(chunk_id="c1", score=0.8)]
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = [2.5]

    with patch("src.retrieval.reranker._get_cross_encoder", return_value=mock_encoder):
        from src.retrieval.reranker import rerank

        result = rerank("query", chunks, top_n=1)

    assert result[0].similarity_score == 2.5


def test_rerank_graceful_degradation_model_unavailable() -> None:
    """When cross-encoder fails to load, return original top_n."""
    chunks = [_chunk(chunk_id="c1"), _chunk(chunk_id="c2")]

    with patch("src.retrieval.reranker._get_cross_encoder", return_value=None):
        from src.retrieval.reranker import rerank

        result = rerank("query", chunks, top_n=1)

    assert len(result) == 1
    assert result[0].chunk_id == "c1"  # original order preserved


def test_rerank_graceful_degradation_predict_failure() -> None:
    """When prediction fails, return original top_n."""
    chunks = [_chunk(chunk_id="c1"), _chunk(chunk_id="c2")]
    mock_encoder = MagicMock()
    mock_encoder.predict.side_effect = RuntimeError("CUDA OOM")

    with patch("src.retrieval.reranker._get_cross_encoder", return_value=mock_encoder):
        from src.retrieval.reranker import rerank

        result = rerank("query", chunks, top_n=2)

    assert len(result) == 2


def test_rerank_empty_chunks() -> None:
    """Empty input returns empty output."""
    from src.retrieval.reranker import rerank

    assert rerank("query", [], top_n=5) == []


def test_rerank_preserves_metadata() -> None:
    """Reranked results should preserve all fields except similarity_score."""
    chunk = _RetrievalResult(
        text="important text",
        chunk_id="c42",
        document_id="manual_dict",
        page_number=7,
        section_title="Section A",
        similarity_score=0.8,
        source_file="doc.pdf",
    )
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = [1.5]

    with patch("src.retrieval.reranker._get_cross_encoder", return_value=mock_encoder):
        from src.retrieval.reranker import rerank

        result = rerank("query", [chunk], top_n=1)

    assert result[0].chunk_id == "c42"
    assert result[0].document_id == "manual_dict"
    assert result[0].page_number == 7
    assert result[0].section_title == "Section A"
    assert result[0].source_file == "doc.pdf"
    assert result[0].text == "important text"
    assert result[0].similarity_score == 1.5  # updated to cross-encoder score
