"""Unit tests for chunk dataset serialization and loading."""

import json
import tempfile
from pathlib import Path

import pytest

from src.chunking.loader import load_chunks_jsonl
from src.chunking.models import Chunk
from src.chunking.serializer import (
    chunk_to_record,
    save_chunks_jsonl,
    validate_chunk_dataset,
    validate_chunk_record,
)
from src.chunking.structural_segmenter import segment_page
from src.chunking.token_chunker import chunk_segment
from src.ingestion.models import Page


def _make_chunk() -> Chunk:
    """Create a sample Chunk for testing."""
    return Chunk(
        chunk_id="doc_p1_s0_c0",
        document_id="doc",
        page_number=1,
        segment_index=0,
        chunk_index=0,
        section_title="1 Chaves",
        article_numbers=["Art. 1º"],
        source_file="doc.pdf",
        text="Sample chunk text.",
        token_count=4,
        char_start=0,
        char_end=18,
    )


def test_chunk_to_record_preserves_all_fields() -> None:
    """Chunk → dict preserves all fields."""
    chunk = _make_chunk()
    record = chunk_to_record(chunk)

    assert record["chunk_id"] == "doc_p1_s0_c0"
    assert record["document_id"] == "doc"
    assert record["page_number"] == 1
    assert record["segment_index"] == 0
    assert record["chunk_index"] == 0
    assert record["section_title"] == "1 Chaves"
    assert record["article_numbers"] == ["Art. 1º"]
    assert record["source_file"] == "doc.pdf"
    assert record["text"] == "Sample chunk text."
    assert record["token_count"] == 4
    assert record["char_start"] == 0
    assert record["char_end"] == 18


def test_save_chunks_jsonl_creates_file() -> None:
    """save_chunks_jsonl creates JSONL file with correct content."""
    chunks = [_make_chunk()]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "corpus_chunks.jsonl"
        count = save_chunks_jsonl(chunks, path)

        assert count == 1
        assert path.exists()

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["chunk_id"] == "doc_p1_s0_c0"
        assert record["text"] == "Sample chunk text."


def test_save_chunks_jsonl_streaming() -> None:
    """save_chunks_jsonl supports iterable (streaming) input."""
    def chunk_gen():
        yield _make_chunk()
        yield Chunk(
            chunk_id="doc_p1_s0_c1",
            document_id="doc",
            page_number=1,
            segment_index=0,
            chunk_index=1,
            section_title="1 Chaves",
            article_numbers=[],
            source_file="doc.pdf",
            text="Second chunk.",
            token_count=2,
        )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "chunks.jsonl"
        count = save_chunks_jsonl(chunk_gen(), path)
        assert count == 2
        with open(path, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 2


def test_validate_chunk_record_valid() -> None:
    """validate_chunk_record passes for valid record."""
    record = chunk_to_record(_make_chunk())
    validate_chunk_record(record)
    validate_chunk_record(record, chunk_size=500)


def test_validate_chunk_record_missing_field() -> None:
    """validate_chunk_record raises for missing required field."""
    record = {"chunk_id": "x", "text": "a", "token_count": 1}
    with pytest.raises(ValueError, match="Missing required"):
        validate_chunk_record(record)


def test_validate_chunk_record_empty_text() -> None:
    """validate_chunk_record raises for empty text."""
    record = chunk_to_record(_make_chunk())
    record["text"] = "   "
    with pytest.raises(ValueError, match="must not be empty"):
        validate_chunk_record(record)


def test_validate_chunk_record_token_limit() -> None:
    """validate_chunk_record raises when token_count exceeds chunk_size."""
    record = chunk_to_record(_make_chunk())
    record["token_count"] = 600
    validate_chunk_record(record)  # no chunk_size, passes
    with pytest.raises(ValueError, match="exceeds chunk_size"):
        validate_chunk_record(record, chunk_size=500)


def test_validate_chunk_record_invalid_chunk_id() -> None:
    """validate_chunk_record raises for invalid chunk_id format."""
    record = chunk_to_record(_make_chunk())
    record["chunk_id"] = "invalid_format"
    with pytest.raises(ValueError, match="chunk_id"):
        validate_chunk_record(record)


def test_validate_chunk_dataset_valid() -> None:
    """validate_chunk_dataset passes for valid dataset."""
    chunks = [_make_chunk()]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "chunks.jsonl"
        save_chunks_jsonl(chunks, path)
        count, errors = validate_chunk_dataset(path)
        assert count == 1
        assert errors == []


def test_validate_chunk_dataset_invalid_json() -> None:
    """validate_chunk_dataset reports invalid JSON lines."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "chunks.jsonl"
        path.write_text("not json\n", encoding="utf-8")
        count, errors = validate_chunk_dataset(path)
        assert count == 0
        assert len(errors) == 1
        assert "Invalid JSON" in errors[0]


def test_load_chunks_jsonl_returns_chunks() -> None:
    """load_chunks_jsonl yields valid Chunk objects."""
    chunks = [_make_chunk()]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "chunks.jsonl"
        save_chunks_jsonl(chunks, path)

        loaded = list(load_chunks_jsonl(path))
        assert len(loaded) == 1
        assert isinstance(loaded[0], Chunk)
        assert loaded[0].chunk_id == "doc_p1_s0_c0"
        assert loaded[0].text == "Sample chunk text."


def test_load_chunks_jsonl_preserves_metadata() -> None:
    """load_chunks_jsonl preserves all metadata."""
    chunk = _make_chunk()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "chunks.jsonl"
        save_chunks_jsonl([chunk], path)
        loaded = next(load_chunks_jsonl(path))

    assert loaded.document_id == chunk.document_id
    assert loaded.page_number == chunk.page_number
    assert loaded.section_title == chunk.section_title
    assert loaded.article_numbers == chunk.article_numbers
    assert loaded.source_file == chunk.source_file


def test_deterministic_dataset() -> None:
    """Same input produces identical chunk_ids and record count."""
    page = Page(
        document_id="doc",
        page_number=1,
        text="Short page text for chunking.",
        source_file=Path("doc.pdf"),
        section_title="Section",
        article_numbers=[],
    )
    segments1 = segment_page(page)
    segments2 = segment_page(page)
    assert len(segments1) == len(segments2)

    chunks1 = []
    chunks2 = []
    for seg in segments1:
        chunks1.extend(chunk_segment(seg))
    for seg in segments2:
        chunks2.extend(chunk_segment(seg))

    assert len(chunks1) == len(chunks2)
    assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]
