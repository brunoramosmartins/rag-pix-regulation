"""Data models representing parsed regulatory documents."""

from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field, ConfigDict, model_validator


class Page(BaseModel):
    """Represents a single page extracted from a PDF document."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(
        ...,
        description="Stable identifier derived from source_file.stem for tracing and citation.",
    )
    page_number: int = Field(
        ..., ge=1, description="1-based page index within the source document."
    )
    text: str = Field(..., description="Extracted textual content of the page.")
    source_file: Path = Field(..., description="Path to the source PDF file.")
    section_title: str | None = Field(
        default=None, description="Section title when detected (e.g. '1 Chaves Pix')."
    )
    article_numbers: list[str] = Field(
        default_factory=list,
        description="Article markers found on page (e.g. ['Art. 1º', '§2º']).",
    )


class Document(BaseModel):
    """
    Represents a full document composed of multiple pages.

    The document preserves page-level segmentation to support downstream
    tasks such as chunking, retrieval attribution, and citation generation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_file: Path = Field(..., description="Path to the source PDF file.")
    pages: list[Page] = Field(
        default_factory=list, description="List of extracted pages."
    )
    total_pages: int = Field(
        ..., ge=0, description="Total number of pages in the document."
    )

    @model_validator(mode="after")
    def validate_page_consistency(self):
        """Ensure metadata consistency between pages and total_pages."""
        if self.pages and self.total_pages != len(self.pages):
            raise ValueError(
                "total_pages must match the number of Page objects provided."
            )
        return self

    @property
    def full_text(self) -> str:
        """Concatenate the text of all pages."""
        return "\n\n".join(page.text for page in self.pages)

    def iter_pages(self) -> Iterable[Page]:
        """Iterate over document pages."""
        yield from self.pages
