"""Chunking module - Document splitting strategies for RAG."""

from .models import StructuralSegment
from .structural_segmenter import (
    segment_document,
    segment_page,
    segment_records,
)

__all__ = [
    "StructuralSegment",
    "segment_page",
    "segment_document",
    "segment_records",
]
