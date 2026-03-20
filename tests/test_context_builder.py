"""Unit tests for RAG context builder."""

from dataclasses import dataclass

from src.rag.context_builder import build_context


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


def test_build_context_concatenates_chunks() -> None:
    """Context joins chunk texts with markers."""
    chunks: list[_RetrievalResult] = [
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
    chunks = [_chunk(f"Chunk {i}.", page=i) for i in range(1, 8)]
    ctx = build_context(chunks, max_chunks=3)
    assert "Chunk 1." in ctx
    assert "Chunk 2." in ctx
    assert "Chunk 3." in ctx
    assert "Chunk 4." not in ctx


def test_build_context_truncates_by_tokens() -> None:
    """When over max_tokens, truncates oldest chunks."""
    # ~4 chars per token, so 100 chars ≈ 25 tokens
    long_text = "x" * 200  # ~50 tokens
    chunks = [
        _chunk("short", page=1),
        _chunk(long_text, page=2),
    ]
    ctx = build_context(chunks, max_chunks=5, max_tokens=30)
    # Should fit "short" + marker + some of long_text, or truncate
    assert len(ctx) <= 30 * 4 + 50  # rough upper bound


def test_build_context_no_truncation_when_under_limit() -> None:
    """Full context when under max_tokens."""
    chunks = [_chunk("A"), _chunk("B")]
    ctx = build_context(chunks, max_chunks=5, max_tokens=1000)
    assert "A" in ctx and "B" in ctx
