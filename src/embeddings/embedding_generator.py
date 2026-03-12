"""Embedding generation pipeline for regulatory chunks."""

from pathlib import Path
from typing import Iterator

from sentence_transformers import SentenceTransformer

from src.chunking.loader import load_chunks_jsonl
from src.chunking.models import Chunk

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_BATCH_SIZE = 32

_model: SentenceTransformer | None = None


def get_embedding_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Return embedding model instance (cached)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def generate_embeddings(
    chunks: list[Chunk],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[tuple[Chunk, list[float]]]:
    """
    Generate embeddings for chunk texts.

    Returns list of (chunk, embedding) tuples. Deterministic for same input.
    """
    if not chunks:
        return []

    model = get_embedding_model(model_name)
    texts = [c.text for c in chunks]

    # NOTE: normalize_embeddings=False here is intentional.
    # Weaviate uses cosine distance natively and handles un-normalized vectors correctly.
    # Query embeddings in query_embedding.py use normalize=True for direct dot-product
    # similarity, but cosine distance is invariant to L2 norm, so results are equivalent.
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=False,
    )

    # Convert numpy array to list of lists for serialization
    result: list[tuple[Chunk, list[float]]] = []
    for chunk, emb in zip(chunks, embeddings, strict=True):
        result.append((chunk, emb.tolist()))

    return result


def generate_embeddings_from_dataset(
    chunks_path: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[tuple[Chunk, list[float]]]:
    """
    Stream embeddings from chunk dataset.

    Loads chunks in batches, generates embeddings, yields (chunk, embedding).
    Memory-efficient for large corpora.
    """
    get_embedding_model(model_name)  # Preload model
    batch: list[Chunk] = []

    for chunk in load_chunks_jsonl(chunks_path):
        batch.append(chunk)
        if len(batch) >= batch_size:
            for pair in generate_embeddings(batch, model_name, batch_size):
                yield pair
            batch = []

    if batch:
        for pair in generate_embeddings(batch, model_name, batch_size):
            yield pair
