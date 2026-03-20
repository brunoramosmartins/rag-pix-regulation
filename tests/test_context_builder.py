"""Unit tests for RAG context builder."""

from dataclasses import dataclass
from unittest.mock import patch


@dataclass
class _RetrievalResult:
    """Minimal RetrievalResult for tests (avoids heavy retrieval imports)."""
    text: str
    chunk_id: str
    document_id: str
    page_number: int
    section_title: str | None
    similarity_score: float | None
    source_file: str | None = None


def _chunk(text: str, doc: str = "manual_dict", page: int = 1) -> _RetrievalResult:
    return _RetrievalResult(
        text=text,
        chunk_id=f"{doc}_p{page}_s0_c0",
        document_id=doc,
        page_number=page,
        section_title=None,
        similarity_score=0.9,
    )


def _mock_count(text: str, model_name: str = "BAAI/bge-m3") -> int:
    """Simple mock: 1 token per character for predictable tests."""
    return len(text)


def test_build_context_concatenates_chunks() -> None:
    """Context joins chunk texts with markers."""
    with patch("src.rag.context_builder.count_tokens", side_effect=_mock_count):
        from src.rag.context_builder import build_context

        chunks = [
            _chunk("First chunk.", doc="d1", page=1),
            _chunk("Second chunk.", doc="d1", page=2),
        ]
        ctx = build_context(chunks, max_chunks=5)
        assert "[d1, p. 1]" in ctx
        assert "First chunk." in ctx
        assert "[d1, p. 2]" in ctx
        assert "Second chunk." in ctx


def test_build_context_respects_max_chunks() -> None:
    """Only first max_chunks are included."""
    with patch("src.rag.context_builder.count_tokens", side_effect=_mock_count):
        from src.rag.context_builder import build_context

        chunks = [_chunk(f"Chunk {i}.", page=i) for i in range(1, 8)]
        ctx = build_context(chunks, max_chunks=3)
        assert "Chunk 1." in ctx
        assert "Chunk 2." in ctx
        assert "Chunk 3." in ctx
        assert "Chunk 4." not in ctx


def test_build_context_greedy_packing_stops_at_budget() -> None:
    """Greedy packing stops when next chunk would exceed token budget."""
    with patch("src.rag.context_builder.count_tokens", side_effect=_mock_count):
        from src.rag.context_builder import build_context

        # Each formatted chunk: "[manual_dict, p. N]\ntext" + separator "\n\n"
        # With mock count, tokens = len(formatted_text)
        chunks = [
            _chunk("A" * 50, page=1),   # ~70 chars with marker
            _chunk("B" * 50, page=2),   # ~70 chars with marker
            _chunk("C" * 50, page=3),   # ~70 chars with marker
        ]
        # Budget that fits 2 chunks but not 3
        ctx = build_context(chunks, max_tokens=160)
        assert "A" * 50 in ctx
        assert "B" * 50 in ctx
        assert "C" * 50 not in ctx


def test_build_context_skips_chunk_exceeding_budget() -> None:
    """When a single chunk exceeds the entire budget, no chunks are included."""
    with patch("src.rag.context_builder.count_tokens", side_effect=_mock_count):
        from src.rag.context_builder import build_context

        chunks = [_chunk("X" * 500, page=1)]
        ctx = build_context(chunks, max_tokens=10)
        assert ctx == ""


def test_build_context_no_truncation_when_under_limit() -> None:
    """Full context when under max_tokens."""
    with patch("src.rag.context_builder.count_tokens", side_effect=_mock_count):
        from src.rag.context_builder import build_context

        chunks = [_chunk("A"), _chunk("B")]
        ctx = build_context(chunks, max_chunks=5, max_tokens=1000)
        assert "A" in ctx and "B" in ctx


def test_build_context_empty_chunks() -> None:
    """Empty chunks list returns empty string."""
    with patch("src.rag.context_builder.count_tokens", side_effect=_mock_count):
        from src.rag.context_builder import build_context

        assert build_context([], max_tokens=100) == ""
