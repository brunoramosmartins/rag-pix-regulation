"""Tests for content hashing and incremental indexing classification."""

import hashlib

from src.chunking.models import Chunk, StructuralSegment, compute_content_hash
from src.chunking.token_chunker import chunk_segment


# ── compute_content_hash ────────────────────────────────────────────────────


def test_hash_deterministic() -> None:
    """Same text always produces the same hash."""
    text = "Art. 1º O arranjo de pagamentos Pix"
    assert compute_content_hash(text) == compute_content_hash(text)


def test_hash_differs_for_different_text() -> None:
    """Different text produces different hashes."""
    h1 = compute_content_hash("Text A")
    h2 = compute_content_hash("Text B")
    assert h1 != h2


def test_hash_is_sha256_hex() -> None:
    """Hash should be a 64-char hex string (SHA-256)."""
    h = compute_content_hash("test")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_hash_matches_stdlib() -> None:
    """Verify hash matches direct hashlib computation."""
    text = "Regulamentação PIX"
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert compute_content_hash(text) == expected


def test_hash_unicode_handling() -> None:
    """Unicode text (accents, special chars) is hashed correctly."""
    text = "§ 2º — não será permitido"
    h = compute_content_hash(text)
    assert len(h) == 64


# ── Chunk model with content_hash ───────────────────────────────────────────


def test_chunk_model_includes_content_hash() -> None:
    """Chunk model has content_hash field."""
    chunk = Chunk(
        chunk_id="doc_p1_s0_c0",
        document_id="doc",
        page_number=1,
        segment_index=0,
        chunk_index=0,
        source_file="doc.pdf",
        text="Test text",
        token_count=2,
        content_hash=compute_content_hash("Test text"),
    )
    assert chunk.content_hash == compute_content_hash("Test text")


def test_chunk_model_default_hash_is_empty() -> None:
    """Backward compatibility: content_hash defaults to empty string."""
    chunk = Chunk(
        chunk_id="doc_p1_s0_c0",
        document_id="doc",
        page_number=1,
        segment_index=0,
        chunk_index=0,
        source_file="doc.pdf",
        text="Test text",
        token_count=2,
    )
    assert chunk.content_hash == ""


def test_chunk_model_dump_includes_hash() -> None:
    """model_dump() includes content_hash for JSONL serialization."""
    chunk = Chunk(
        chunk_id="doc_p1_s0_c0",
        document_id="doc",
        page_number=1,
        segment_index=0,
        chunk_index=0,
        source_file="doc.pdf",
        text="Test text",
        token_count=2,
        content_hash="abc123",
    )
    data = chunk.model_dump()
    assert "content_hash" in data
    assert data["content_hash"] == "abc123"


# ── token_chunker produces content_hash ─────────────────────────────────────


def test_chunk_segment_produces_hash() -> None:
    """chunk_segment() populates content_hash on each chunk."""
    segment = StructuralSegment(
        document_id="doc",
        page_number=1,
        segment_index=0,
        section_title=None,
        article_numbers=[],
        source_file="doc.pdf",
        text="Short text for hashing.",
    )
    chunks = chunk_segment(segment, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].content_hash != ""
    assert len(chunks[0].content_hash) == 64


def test_chunk_segment_hash_matches_text() -> None:
    """content_hash corresponds to the actual chunk text."""
    segment = StructuralSegment(
        document_id="doc",
        page_number=1,
        segment_index=0,
        section_title=None,
        article_numbers=[],
        source_file="doc.pdf",
        text="Exact text for verification.",
    )
    chunks = chunk_segment(segment, chunk_size=500, chunk_overlap=50)
    expected_hash = compute_content_hash(chunks[0].text)
    assert chunks[0].content_hash == expected_hash


def test_chunk_segment_multi_chunk_all_have_hash() -> None:
    """Multi-chunk segment: every chunk gets a content_hash."""
    long_text = "word " * 600
    segment = StructuralSegment(
        document_id="doc",
        page_number=1,
        segment_index=0,
        section_title=None,
        article_numbers=[],
        source_file="doc.pdf",
        text=long_text,
    )
    chunks = chunk_segment(segment, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.content_hash != ""
        assert chunk.content_hash == compute_content_hash(chunk.text)


# ── Incremental classification logic ────────────────────────────────────────


def _classify_chunks_pure(chunks, existing_hashes):
    """Pure reimplementation of indexer._classify_chunks for testing without heavy imports."""
    new, changed, unchanged = [], [], []
    for chunk in chunks:
        old_hash = existing_hashes.get(chunk.chunk_id)
        if old_hash is None:
            new.append(chunk)
        elif old_hash != chunk.content_hash:
            changed.append(chunk)
        else:
            unchanged.append(chunk)
    return new, changed, unchanged


def test_classify_all_new() -> None:
    """When no existing hashes, all chunks are classified as new."""

    chunks = [
        Chunk(
            chunk_id=f"doc_p1_s0_c{i}",
            document_id="doc",
            page_number=1,
            segment_index=0,
            chunk_index=i,
            source_file="doc.pdf",
            text=f"text {i}",
            token_count=2,
            content_hash=compute_content_hash(f"text {i}"),
        )
        for i in range(3)
    ]
    new, changed, unchanged = _classify_chunks_pure(chunks, {})
    assert len(new) == 3
    assert len(changed) == 0
    assert len(unchanged) == 0


def test_classify_all_unchanged() -> None:
    """When all hashes match, all chunks are skipped."""
    chunks = [
        Chunk(
            chunk_id="doc_p1_s0_c0",
            document_id="doc",
            page_number=1,
            segment_index=0,
            chunk_index=0,
            source_file="doc.pdf",
            text="same text",
            token_count=2,
            content_hash=compute_content_hash("same text"),
        )
    ]
    existing = {"doc_p1_s0_c0": compute_content_hash("same text")}
    new, changed, unchanged = _classify_chunks_pure(chunks, existing)
    assert len(new) == 0
    assert len(changed) == 0
    assert len(unchanged) == 1


def test_classify_changed() -> None:
    """When hash differs, chunk is classified as changed."""
    chunks = [
        Chunk(
            chunk_id="doc_p1_s0_c0",
            document_id="doc",
            page_number=1,
            segment_index=0,
            chunk_index=0,
            source_file="doc.pdf",
            text="new text",
            token_count=2,
            content_hash=compute_content_hash("new text"),
        )
    ]
    existing = {"doc_p1_s0_c0": compute_content_hash("old text")}
    new, changed, unchanged = _classify_chunks_pure(chunks, existing)
    assert len(new) == 0
    assert len(changed) == 1
    assert len(unchanged) == 0


def test_classify_mixed() -> None:
    """Mixed scenario: some new, some changed, some unchanged."""
    hash_a = compute_content_hash("text a")
    hash_b_old = compute_content_hash("text b old")
    hash_b_new = compute_content_hash("text b new")
    hash_c = compute_content_hash("text c")

    chunks = [
        Chunk(
            chunk_id="doc_p1_s0_c0",
            document_id="doc",
            page_number=1,
            segment_index=0,
            chunk_index=0,
            source_file="doc.pdf",
            text="text a",
            token_count=2,
            content_hash=hash_a,
        ),
        Chunk(
            chunk_id="doc_p1_s0_c1",
            document_id="doc",
            page_number=1,
            segment_index=0,
            chunk_index=1,
            source_file="doc.pdf",
            text="text b new",
            token_count=2,
            content_hash=hash_b_new,
        ),
        Chunk(
            chunk_id="doc_p1_s0_c2",
            document_id="doc",
            page_number=1,
            segment_index=0,
            chunk_index=2,
            source_file="doc.pdf",
            text="text c",
            token_count=2,
            content_hash=hash_c,
        ),
    ]
    existing = {
        # c0 unchanged
        "doc_p1_s0_c0": hash_a,
        # c1 changed
        "doc_p1_s0_c1": hash_b_old,
        # c2 not in existing → new
    }
    new, changed, unchanged = _classify_chunks_pure(chunks, existing)
    assert len(new) == 1
    assert new[0].chunk_id == "doc_p1_s0_c2"
    assert len(changed) == 1
    assert changed[0].chunk_id == "doc_p1_s0_c1"
    assert len(unchanged) == 1
    assert unchanged[0].chunk_id == "doc_p1_s0_c0"
