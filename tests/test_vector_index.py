"""Unit tests for Weaviate vector indexing."""

import pytest

from src.vectorstore.weaviate_client import (
    chunk_to_weaviate_properties,
    init_chunk_collection,
    is_weaviate_ready,
)
from src.chunking.models import Chunk


def _make_chunk() -> Chunk:
    """Create a minimal Chunk for testing."""
    return Chunk(
        chunk_id="doc_p1_s0_c0",
        document_id="doc",
        page_number=1,
        segment_index=0,
        chunk_index=0,
        section_title="1 Chaves",
        article_numbers=["Art. 1º"],
        source_file="doc.pdf",
        text="Sample text.",
        token_count=4,
    )


def test_chunk_to_weaviate_properties() -> None:
    """chunk_to_weaviate_properties converts Chunk to dict."""
    chunk = _make_chunk()
    props = chunk_to_weaviate_properties(chunk)
    assert props["chunk_id"] == "doc_p1_s0_c0"
    assert props["document_id"] == "doc"
    assert props["page_number"] == 1
    assert props["section_title"] == "1 Chaves"
    assert props["article_numbers"] == ["Art. 1º"]
    assert props["text"] == "Sample text."


def test_chunk_to_weaviate_properties_none_section() -> None:
    """chunk_to_weaviate_properties handles None section_title."""
    chunk = _make_chunk()
    chunk.section_title = None
    props = chunk_to_weaviate_properties(chunk)
    assert props["section_title"] == ""


@pytest.mark.skipif(
    not is_weaviate_ready(),
    reason="Weaviate not running",
)
def test_init_chunk_collection() -> None:
    """init_chunk_collection creates collection when Weaviate is available."""
    from src.vectorstore.weaviate_client import get_weaviate_client

    client = get_weaviate_client()
    init_chunk_collection(client, recreate=True)
    assert client.collections.exists("Chunk")
