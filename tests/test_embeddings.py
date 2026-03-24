"""Unit tests for embedding generation and validation."""

import tempfile
from pathlib import Path

import pytest

from src.chunking.models import Chunk
from src.chunking.serializer import save_chunks_jsonl
from src.embeddings.embedding_generator import (
    generate_embeddings,
    generate_embeddings_from_dataset,
    get_embedding_model,
)
from src.embeddings.validation import (
    BGE_M3_DIMENSIONS,
    validate_embedding,
    validate_embeddings_batch,
    validate_chunk_embedding_pairs,
)

pytestmark = pytest.mark.integration


def _make_chunk(text: str = "Sample regulatory text.") -> Chunk:
    """Create a minimal Chunk for testing."""
    return Chunk(
        chunk_id="doc_p1_s0_c0",
        document_id="doc",
        page_number=1,
        segment_index=0,
        chunk_index=0,
        section_title="1 Chaves",
        article_numbers=[],
        source_file="doc.pdf",
        text=text,
        token_count=4,
    )


def test_get_embedding_model_returns_model() -> None:
    """get_embedding_model returns SentenceTransformer instance."""
    model = get_embedding_model()
    assert model is not None
    assert hasattr(model, "encode")


def test_generate_embeddings_dimensionality() -> None:
    """Generated embeddings have expected dimensionality (1024)."""
    chunks = [_make_chunk()]
    pairs = generate_embeddings(chunks)
    assert len(pairs) == 1
    _, embedding = pairs[0]
    assert len(embedding) == BGE_M3_DIMENSIONS


def test_generate_embeddings_deterministic() -> None:
    """Same input produces same embeddings."""
    chunks = [_make_chunk("Deterministic text.")]
    pairs1 = generate_embeddings(chunks)
    pairs2 = generate_embeddings(chunks)
    assert pairs1[0][1] == pairs2[0][1]


def test_generate_embeddings_batch() -> None:
    """Multiple chunks produce multiple embeddings."""
    chunks = [_make_chunk(f"Text {i}.") for i in range(5)]
    pairs = generate_embeddings(chunks, batch_size=2)
    assert len(pairs) == 5
    for chunk, emb in pairs:
        assert len(emb) == BGE_M3_DIMENSIONS


def test_generate_embeddings_preserves_chunk() -> None:
    """Embeddings are correctly associated with chunks."""
    chunk = _make_chunk("Unique content.")
    pairs = generate_embeddings([chunk])
    assert pairs[0][0].chunk_id == chunk.chunk_id
    assert pairs[0][0].text == chunk.text


def test_generate_embeddings_from_dataset() -> None:
    """generate_embeddings_from_dataset yields from JSONL file."""
    chunks = [_make_chunk("Dataset text.")]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "chunks.jsonl"
        save_chunks_jsonl(chunks, path)
        results = list(generate_embeddings_from_dataset(path, batch_size=1))
    assert len(results) == 1
    _, emb = results[0]
    assert len(emb) == BGE_M3_DIMENSIONS


def test_validate_embedding_valid() -> None:
    """validate_embedding passes for valid vector."""
    emb = [0.1] * BGE_M3_DIMENSIONS
    validate_embedding(emb)


def test_validate_embedding_wrong_dimension() -> None:
    """validate_embedding raises for wrong dimension."""
    emb = [0.1] * 512
    with pytest.raises(ValueError, match="dimension"):
        validate_embedding(emb)


def test_validate_embedding_none() -> None:
    """validate_embedding raises for None."""
    with pytest.raises(ValueError, match="None"):
        validate_embedding(None)


def test_validate_embedding_nan() -> None:
    """validate_embedding raises for NaN."""
    emb = [0.1] * BGE_M3_DIMENSIONS
    emb[0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_embedding(emb)


def test_validate_embeddings_batch_valid() -> None:
    """validate_embeddings_batch passes for valid batch."""
    embs = [[0.1] * BGE_M3_DIMENSIONS for _ in range(3)]
    validate_embeddings_batch(embs)


def test_validate_chunk_embedding_pairs_valid() -> None:
    """validate_chunk_embedding_pairs passes when all have embeddings."""
    chunk = _make_chunk()
    pairs = [(chunk, [0.1] * BGE_M3_DIMENSIONS)]
    validate_chunk_embedding_pairs(pairs)


def test_validate_chunk_embedding_pairs_missing() -> None:
    """validate_chunk_embedding_pairs raises when chunk has no embedding."""
    chunk = _make_chunk()
    pairs = [(chunk, None)]
    with pytest.raises(ValueError, match="no embedding"):
        validate_chunk_embedding_pairs(pairs)
