"""Structure-aware segmentation of regulatory document pages."""

import re
from pathlib import Path
from typing import Iterable

from src.ingestion.models import Document, Page

from .models import StructuralSegment

ART_RE = re.compile(r"^Art\.\s*\d+º?", re.MULTILINE)
PARA_RE = re.compile(r"^§\s*\d+º?", re.MULTILINE)
SECTION_RE = re.compile(r"^\d+(?:\.\d+)+\s+", re.MULTILINE)


def _detect_article_markers(text: str) -> list[re.Match]:
    """Find Art. Xº patterns at line start."""
    return list(ART_RE.finditer(text))


def _detect_paragraph_markers(text: str) -> list[re.Match]:
    """Find §Xº patterns at line start."""
    return list(PARA_RE.finditer(text))


def _detect_section_markers(text: str) -> list[re.Match]:
    """Find numbered subsection patterns (3.1, 4.2.1) at line start."""
    return list(SECTION_RE.finditer(text))


def _is_structural_marker(block: str) -> bool:
    """Check if block starts with a structural marker (Art., §, or subsection number)."""
    stripped = block.strip()
    if not stripped:
        return False
    return bool(
        ART_RE.match(stripped) or PARA_RE.match(stripped) or SECTION_RE.match(stripped)
    )


def _extract_markers_from_segment(text: str) -> list[str]:
    """Extract Art. and § markers present in segment text, in order of appearance."""
    markers: list[str] = []
    seen: set[str] = set()
    for pattern in (ART_RE, PARA_RE):
        for match in pattern.finditer(text):
            marker = match.group(0)
            if marker not in seen:
                seen.add(marker)
                markers.append(marker)
    return markers


def segment_page(page: Page) -> list[StructuralSegment]:
    """
    Split page text into structural segments.

    Uses paragraph blocks (split by \\n\\n) and groups by structural markers
    (Art., §, numbered subsections). Does not modify text.
    """
    blocks = [b.strip() for b in page.text.split("\n\n") if b.strip()]
    if not blocks:
        return [
            StructuralSegment(
                document_id=page.document_id,
                page_number=page.page_number,
                section_title=page.section_title,
                article_numbers=[],
                source_file=page.source_file.name,
                text="",
                segment_index=0,
                char_start=0,
                char_end=0,
            )
        ]

    segments: list[StructuralSegment] = []
    current_blocks: list[str] = []
    segment_index = 0
    char_offset = 0

    for block in blocks:
        if _is_structural_marker(block):
            if current_blocks:
                segment_text = "\n\n".join(current_blocks)
                char_start = char_offset
                char_end = char_offset + len(segment_text)

                segments.append(
                    StructuralSegment(
                        document_id=page.document_id,
                        page_number=page.page_number,
                        section_title=page.section_title,
                        article_numbers=_extract_markers_from_segment(segment_text),
                        source_file=page.source_file.name,
                        text=segment_text,
                        segment_index=segment_index,
                        char_start=char_start,
                        char_end=char_end,
                    )
                )
                segment_index += 1
                char_offset = char_end

            current_blocks = [block]
        else:
            current_blocks.append(block)

    if current_blocks:
        segment_text = "\n\n".join(current_blocks)
        char_start = char_offset
        char_end = char_offset + len(segment_text)

        segments.append(
            StructuralSegment(
                document_id=page.document_id,
                page_number=page.page_number,
                section_title=page.section_title,
                article_numbers=_extract_markers_from_segment(segment_text),
                source_file=page.source_file.name,
                text=segment_text,
                segment_index=segment_index,
                char_start=char_start,
                char_end=char_end,
            )
        )

    return segments


def segment_document(document: Document) -> list[StructuralSegment]:
    """Segment all pages in a document."""
    segments: list[StructuralSegment] = []
    for page in document.pages:
        segments.extend(segment_page(page))
    return segments


def segment_records(records: Iterable[dict]) -> list[dict]:
    """
    Segment JSONL records into structural segments.

    Converts dict → Page → segments → dicts for pipeline integration.
    """
    segments: list[dict] = []
    for rec in records:
        page = Page(
            document_id=rec["document_id"],
            page_number=rec["page_number"],
            text=rec["text"],
            source_file=Path(rec["source_file"]),
            section_title=rec.get("section_title"),
            article_numbers=rec.get("article_numbers", []),
        )
        for seg in segment_page(page):
            segments.append(seg.model_dump())
    return segments
