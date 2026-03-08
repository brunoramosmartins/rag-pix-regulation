"""Unit tests for token-based chunking."""

from src.chunking import chunk_segment, chunk_segments, chunk_records
from src.chunking.models import StructuralSegment
from src.chunking.token_chunker import (
    _count_tokens,
    _generate_chunk_id,
    _get_tokenizer,
)


def test_chunk_small_segment() -> None:
    """Segment with tokens < chunk_size produces 1 chunk."""
    segment = StructuralSegment(
        document_id="doc",
        page_number=1,
        segment_index=0,
        section_title=None,
        article_numbers=[],
        source_file="doc.pdf",
        text="Short text.",
        char_start=0,
        char_end=10,
    )
    chunks = chunk_segment(segment, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].token_count <= 500


def test_chunk_large_segment() -> None:
    """Segment with tokens > chunk_size produces multiple chunks."""
    tokenizer = _get_tokenizer()
    long_text = "word " * 600
    token_count = _count_tokens(long_text, tokenizer)
    assert token_count > 500, f"Need >500 tokens, got {token_count}"

    segment = StructuralSegment(
        document_id="doc",
        page_number=1,
        segment_index=0,
        section_title="1 Chaves",
        article_numbers=[],
        source_file="doc.pdf",
        text=long_text,
    )
    chunks = chunk_segment(segment, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 2
    assert all(c.token_count <= 500 for c in chunks)


def test_chunk_overlap_validation() -> None:
    """chunk_overlap >= chunk_size raises ValueError."""
    segment = StructuralSegment(
        document_id="doc",
        page_number=1,
        segment_index=0,
        section_title=None,
        article_numbers=[],
        source_file="doc.pdf",
        text="Some text.",
    )
    try:
        chunk_segment(segment, chunk_size=100, chunk_overlap=100)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "overlap" in str(e).lower()


def test_chunk_metadata_preserved() -> None:
    """Chunks inherit document_id, page_number, section_title, article_numbers."""
    segment = StructuralSegment(
        document_id="x_manual",
        page_number=5,
        segment_index=2,
        section_title="1 Chaves Pix",
        article_numbers=["Art. 1º"],
        source_file="doc.pdf",
        text="Content here.",
    )
    chunks = chunk_segment(segment)
    assert len(chunks) == 1
    assert chunks[0].document_id == "x_manual"
    assert chunks[0].page_number == 5
    assert chunks[0].segment_index == 2
    assert chunks[0].section_title == "1 Chaves Pix"
    assert chunks[0].article_numbers == ["Art. 1º"]
    assert chunks[0].source_file == "doc.pdf"


def test_chunk_id_deterministic() -> None:
    """Same input produces same chunk_ids."""
    segment = StructuralSegment(
        document_id="doc",
        page_number=1,
        segment_index=0,
        section_title=None,
        article_numbers=[],
        source_file="doc.pdf",
        text="Deterministic text.",
    )
    chunks1 = chunk_segment(segment)
    chunks2 = chunk_segment(segment)
    assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]


def test_chunk_id_format() -> None:
    """chunk_id follows document_id_pN_sN_cN pattern."""
    chunk_id = _generate_chunk_id("doc", 5, 2, 1)
    assert chunk_id == "doc_p5_s2_c1"


def test_chunk_token_limit() -> None:
    """No chunk exceeds chunk_size tokens."""
    long_text = "token " * 400
    segment = StructuralSegment(
        document_id="doc",
        page_number=1,
        segment_index=0,
        section_title=None,
        article_numbers=[],
        source_file="doc.pdf",
        text=long_text,
    )
    chunks = chunk_segment(segment, chunk_size=200, chunk_overlap=20)
    assert all(c.token_count <= 200 for c in chunks)


def test_chunk_empty_segment() -> None:
    """Empty segment returns empty list."""
    segment = StructuralSegment(
        document_id="doc",
        page_number=1,
        segment_index=0,
        section_title=None,
        article_numbers=[],
        source_file="doc.pdf",
        text="   \n\n  ",
    )
    chunks = chunk_segment(segment)
    assert chunks == []


def test_chunk_segments() -> None:
    """chunk_segments processes multiple segments."""
    segments = [
        StructuralSegment(
            document_id="doc",
            page_number=1,
            segment_index=i,
            section_title=None,
            article_numbers=[],
            source_file="doc.pdf",
            text="Short segment.",
        )
        for i in range(3)
    ]
    chunks = chunk_segments(segments)
    assert len(chunks) == 3
    assert chunks[0].segment_index == 0
    assert chunks[1].segment_index == 1
    assert chunks[2].segment_index == 2


def test_chunk_records() -> None:
    """chunk_records converts segment dicts to chunk dicts."""
    records = [
        {
            "document_id": "doc",
            "page_number": 1,
            "segment_index": 0,
            "section_title": None,
            "article_numbers": [],
            "source_file": "doc.pdf",
            "text": "Segment text.",
            "char_start": None,
            "char_end": None,
        }
    ]
    chunks = chunk_records(records)
    assert len(chunks) == 1
    assert chunks[0]["document_id"] == "doc"
    assert "chunk_id" in chunks[0]
    assert "token_count" in chunks[0]
