"""Data models for chunking stage."""

from pydantic import BaseModel, Field


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
