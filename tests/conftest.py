"""Shared pytest fixtures for the RAG Pix Regulation test suite."""

import pytest

from src.chunking.models import Chunk, StructuralSegment


@pytest.fixture
def sample_chunk() -> Chunk:
    """Minimal valid Chunk instance for testing serialization and indexing."""
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


@pytest.fixture
def sample_segment() -> StructuralSegment:
    """Minimal valid StructuralSegment instance for testing."""
    return StructuralSegment(
        document_id="doc",
        page_number=1,
        segment_index=0,
        section_title="1 Chaves Pix",
        article_numbers=["Art. 1º"],
        source_file="doc.pdf",
        text="Conteúdo regulatório de exemplo.",
        char_start=0,
        char_end=32,
    )
