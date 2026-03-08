"""Chunking module - Document splitting strategies for RAG."""

from .loader import load_chunks_jsonl
from .models import Chunk, StructuralSegment
from .serializer import (
    chunk_to_record,
    save_chunks_jsonl,
    validate_chunk_dataset,
    validate_chunk_record,
)
from .structural_segmenter import (
    segment_document,
    segment_page,
    segment_records,
)
from .token_chunker import (
    chunk_records,
    chunk_segment,
    chunk_segments,
)

__all__ = [
    "Chunk",
    "StructuralSegment",
    "segment_page",
    "segment_document",
    "segment_records",
    "chunk_segment",
    "chunk_segments",
    "chunk_records",
    "chunk_to_record",
    "save_chunks_jsonl",
    "load_chunks_jsonl",
    "validate_chunk_record",
    "validate_chunk_dataset",
]
