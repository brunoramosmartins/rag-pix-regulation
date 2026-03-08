"""Data models for chunking stage."""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Final retrieval unit for embedding and vector indexing."""

    chunk_id: str = Field(..., description="Deterministic unique identifier.")
    document_id: str = Field(..., description="Document identifier.")
    page_number: int = Field(..., ge=1, description="1-based page index.")
    segment_index: int = Field(..., ge=0, description="Segment index within page.")
    chunk_index: int = Field(..., ge=0, description="Chunk index within segment.")
    section_title: str | None = Field(default=None, description="Section title from segment.")
    article_numbers: list[str] = Field(
        default_factory=list,
        description="Article/paragraph markers from segment.",
    )
    source_file: str = Field(..., description="Source PDF filename.")
    text: str = Field(..., description="Chunk text content.")
    token_count: int = Field(..., ge=0, description="Number of tokens in chunk.")
    char_start: int | None = Field(default=None, description="Start offset in segment text.")
    char_end: int | None = Field(default=None, description="End offset in segment text.")


class StructuralSegment(BaseModel):
    """A semantic segment of page text, split by structural boundaries."""

    document_id: str = Field(..., description="Document identifier.")
    page_number: int = Field(..., ge=1, description="1-based page index.")
    section_title: str | None = Field(default=None, description="Section title from page.")
    article_numbers: list[str] = Field(
        default_factory=list,
        description="Article/paragraph markers in this segment (e.g. ['Art. 1º', '§2º']).",
    )
    source_file: str = Field(..., description="Source PDF filename.")
    text: str = Field(..., description="Segment text content.")
    segment_index: int = Field(..., ge=0, description="Order within the page.")
    char_start: int | None = Field(default=None, description="Start offset in original page text.")
    char_end: int | None = Field(default=None, description="End offset in original page text.")
