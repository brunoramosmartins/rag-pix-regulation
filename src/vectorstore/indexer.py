"""Chunk indexing pipeline for Weaviate vector database.

Supports incremental indexing: only new or changed chunks are embedded and indexed.
Uses content hashing (SHA-256) to detect changes.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter

from src.chunking.loader import load_chunks_jsonl
from src.chunking.models import Chunk
from src.config import get_settings
from src.embeddings.embedding_generator import generate_embeddings
from src.embeddings.validation import BGE_M3_DIMENSIONS, validate_chunk_embedding_pairs
from src.observability.tracing import trace_span

from .weaviate_client import (
    CHUNK_COLLECTION,
    chunk_to_weaviate_properties,
    get_weaviate_client,
    init_chunk_collection,
)

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100
EMBEDDING_BATCH_SIZE = 32


@dataclass
class IndexingStats:
    """Statistics from an indexing run."""

    total: int = 0
    new: int = 0
    updated: int = 0
    skipped: int = 0
    deleted: int = 0
    embedding_time_ms: float = 0
    insert_time_ms: float = 0
    errors: list[str] = field(default_factory=list)

    @property
    def indexed(self) -> int:
        return self.new + self.updated


def _fetch_existing_hashes(collection, chunk_ids: list[str]) -> dict[str, str]:
    """Fetch content_hash for existing chunks in Weaviate by chunk_id.

    Returns a dict of {chunk_id: content_hash}.
    """
    existing: dict[str, str] = {}
    for cid in chunk_ids:
        try:
            result = collection.query.fetch_objects(
                filters=Filter.by_property("chunk_id").equal(cid),
                limit=1,
                return_properties=["chunk_id", "content_hash"],
            )
            if result.objects:
                obj = result.objects[0]
                existing[cid] = obj.properties.get("content_hash", "")
        except Exception as e:
            logger.debug("Failed to query chunk %s: %s", cid, e)
    return existing


def _delete_chunks_by_ids(collection, chunk_ids: list[str]) -> int:
    """Delete chunks from Weaviate by chunk_id. Returns count deleted."""
    deleted = 0
    for cid in chunk_ids:
        try:
            result = collection.query.fetch_objects(
                filters=Filter.by_property("chunk_id").equal(cid),
                limit=1,
            )
            if result.objects:
                collection.data.delete_by_id(result.objects[0].uuid)
                deleted += 1
        except Exception as e:
            logger.debug("Failed to delete chunk %s: %s", cid, e)
    return deleted


def _classify_chunks(
    chunks: list[Chunk],
    existing_hashes: dict[str, str],
) -> tuple[list[Chunk], list[Chunk], list[Chunk]]:
    """Classify chunks into new, changed, and unchanged.

    Returns (new_chunks, changed_chunks, unchanged_chunks).
    """
    new_chunks: list[Chunk] = []
    changed_chunks: list[Chunk] = []
    unchanged_chunks: list[Chunk] = []

    for chunk in chunks:
        old_hash = existing_hashes.get(chunk.chunk_id)
        if old_hash is None:
            new_chunks.append(chunk)
        elif old_hash != chunk.content_hash:
            changed_chunks.append(chunk)
        else:
            unchanged_chunks.append(chunk)

    return new_chunks, changed_chunks, unchanged_chunks


def index_chunks(
    chunks_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    recreate_collection: bool = False,
) -> int:
    """
    Load chunks, generate embeddings, and index into Weaviate.

    When recreate_collection is False, performs incremental indexing:
    only new or changed chunks (by content_hash) are embedded and inserted.

    Returns the number of chunks indexed (new + updated).
    """
    settings = get_settings()
    embedding_model = settings.embeddings.model

    with trace_span(
        "indexing_pipeline",
        attributes={"indexing.recreate_collection": recreate_collection},
        openinference_span_kind="CHAIN",
    ) as pipeline_span:
        client = get_weaviate_client()
        init_chunk_collection(client, recreate=recreate_collection)
        collection = client.collections.get(CHUNK_COLLECTION)

        chunks = list(load_chunks_jsonl(chunks_path))
        if not chunks:
            logger.warning("No chunks to index")
            return 0

        stats = IndexingStats(total=len(chunks))

        if pipeline_span and pipeline_span.is_recording():
            pipeline_span.set_attribute("indexing.chunks_total", stats.total)
            pipeline_span.set_attribute("indexing.embedding_model", embedding_model)

        # --- Incremental classification ---
        if recreate_collection:
            new_chunks = chunks
            changed_chunks: list[Chunk] = []
        else:
            with trace_span(
                "fetch_existing_hashes",
                attributes={"chunks_to_check": len(chunks)},
            ):
                chunk_ids = [c.chunk_id for c in chunks]
                existing_hashes = _fetch_existing_hashes(collection, chunk_ids)

            new_chunks, changed_chunks, unchanged_chunks = _classify_chunks(
                chunks, existing_hashes
            )
            stats.skipped = len(unchanged_chunks)

            logger.info(
                "Incremental analysis: %d new, %d changed, %d unchanged (skipped)",
                len(new_chunks),
                len(changed_chunks),
                stats.skipped,
            )

        # Delete changed chunks before re-inserting with new embeddings
        if changed_chunks:
            with trace_span(
                "delete_changed_chunks",
                attributes={"chunks_to_delete": len(changed_chunks)},
            ):
                stats.deleted = _delete_chunks_by_ids(
                    collection, [c.chunk_id for c in changed_chunks]
                )

        # Chunks that need embedding: new + changed
        chunks_to_embed = new_chunks + changed_chunks
        if not chunks_to_embed:
            logger.info(
                "All %d chunks unchanged — nothing to index", stats.total
            )
            if pipeline_span and pipeline_span.is_recording():
                pipeline_span.set_attribute("indexing.chunks_skipped", stats.skipped)
                pipeline_span.set_attribute("indexing.chunks_indexed", 0)
            return 0

        # --- Embed and insert ---
        total_indexed = 0
        batch: list[DataObject] = []

        for i in range(0, len(chunks_to_embed), EMBEDDING_BATCH_SIZE):
            chunk_batch = chunks_to_embed[i : i + EMBEDDING_BATCH_SIZE]

            t0 = time.perf_counter()
            with trace_span(
                "batch_embedding",
                attributes={
                    "batch_index": i // EMBEDDING_BATCH_SIZE,
                    "batch_size": len(chunk_batch),
                },
            ):
                pairs = generate_embeddings(
                    chunk_batch, batch_size=EMBEDDING_BATCH_SIZE
                )
                validate_chunk_embedding_pairs(pairs, expected_dim=BGE_M3_DIMENSIONS)
            stats.embedding_time_ms += (time.perf_counter() - t0) * 1000

            for chunk, embedding in pairs:
                props = chunk_to_weaviate_properties(
                    chunk, embedding_model=embedding_model
                )
                batch.append(DataObject(properties=props, vector=embedding))

            if len(batch) >= batch_size:
                t1 = time.perf_counter()
                with trace_span(
                    "weaviate_insert",
                    attributes={"batch_size": len(batch)},
                ):
                    collection.data.insert_many(batch)
                stats.insert_time_ms += (time.perf_counter() - t1) * 1000
                total_indexed += len(batch)
                logger.info(
                    "Indexed %d / %d chunks", total_indexed, len(chunks_to_embed)
                )
                batch = []

        if batch:
            t1 = time.perf_counter()
            with trace_span(
                "weaviate_insert",
                attributes={"batch_size": len(batch)},
            ):
                collection.data.insert_many(batch)
            stats.insert_time_ms += (time.perf_counter() - t1) * 1000
            total_indexed += len(batch)
            logger.info(
                "Indexed %d / %d chunks", total_indexed, len(chunks_to_embed)
            )

        # Classify new vs updated counts
        stats.new = min(len(new_chunks), total_indexed)
        stats.updated = total_indexed - stats.new

        logger.info(
            "Indexing complete: %d new, %d updated, %d skipped "
            "(embedding: %.0fms, insert: %.0fms)",
            stats.new,
            stats.updated,
            stats.skipped,
            stats.embedding_time_ms,
            stats.insert_time_ms,
        )

        if pipeline_span and pipeline_span.is_recording():
            pipeline_span.set_attribute("indexing.chunks_new", stats.new)
            pipeline_span.set_attribute("indexing.chunks_updated", stats.updated)
            pipeline_span.set_attribute("indexing.chunks_skipped", stats.skipped)
            pipeline_span.set_attribute("indexing.chunks_indexed", total_indexed)
            pipeline_span.set_attribute(
                "indexing.embedding_time_ms", stats.embedding_time_ms
            )
            pipeline_span.set_attribute(
                "indexing.insert_time_ms", stats.insert_time_ms
            )

        return total_indexed
