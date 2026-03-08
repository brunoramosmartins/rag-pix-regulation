"""Chunk indexing pipeline for Weaviate vector database."""

import logging
from pathlib import Path

from weaviate.classes.data import DataObject

from src.chunking.loader import load_chunks_jsonl
from src.embeddings.embedding_generator import generate_embeddings
from src.embeddings.validation import BGE_M3_DIMENSIONS, validate_chunk_embedding_pairs

from .weaviate_client import (
    CHUNK_COLLECTION,
    chunk_to_weaviate_properties,
    get_weaviate_client,
    init_chunk_collection,
)

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100
EMBEDDING_BATCH_SIZE = 32


def index_chunks(
    chunks_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    recreate_collection: bool = False,
) -> int:
    """
    Load chunks, generate embeddings, and index into Weaviate.

    Returns the number of chunks indexed.
    """
    client = get_weaviate_client()
    init_chunk_collection(client, recreate=recreate_collection)
    collection = client.collections.get(CHUNK_COLLECTION)

    chunks = list(load_chunks_jsonl(chunks_path))
    if not chunks:
        logger.warning("No chunks to index")
        return 0

    total_chunks = len(chunks)
    total_indexed = 0
    batch: list[tuple] = []

    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        chunk_batch = chunks[i : i + EMBEDDING_BATCH_SIZE]
        pairs = generate_embeddings(chunk_batch, batch_size=EMBEDDING_BATCH_SIZE)
        validate_chunk_embedding_pairs(pairs, expected_dim=BGE_M3_DIMENSIONS)

        for chunk, embedding in pairs:
            props = chunk_to_weaviate_properties(chunk)
            batch.append(
                DataObject(
                    properties=props,
                    vector=embedding,
                )
            )

        if len(batch) >= batch_size:
            collection.data.insert_many(batch)
            total_indexed += len(batch)
            logger.info("Indexed %d / %d chunks", total_indexed, total_chunks)
            batch = []

    if batch:
        collection.data.insert_many(batch)
        total_indexed += len(batch)
        logger.info("Indexed %d / %d chunks", total_indexed, total_chunks)

    return total_indexed
