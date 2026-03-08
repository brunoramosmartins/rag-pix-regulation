"""Unit tests for structural segmentation."""

from pathlib import Path

from src.chunking import segment_page, segment_document, segment_records
from src.chunking.structural_segmenter import _is_structural_marker
from src.ingestion.models import Document, Page


def test_segment_page_with_articles() -> None:
    """Page with Art. 1º and Art. 2º produces 2 segments."""
    page = Page(
        document_id="doc",
        page_number=1,
        text="Art. 1º First article content.\n\nArt. 2º Second article content.",
        source_file=Path("doc.pdf"),
    )
    segments = segment_page(page)
    assert len(segments) == 2
    assert "Art. 1º" in segments[0].article_numbers
    assert "Art. 2º" in segments[1].article_numbers
    assert segments[0].segment_index == 0
    assert segments[1].segment_index == 1


def test_segment_page_with_paragraphs() -> None:
    """Page with §1º and §2º produces 2 segments."""
    page = Page(
        document_id="doc",
        page_number=1,
        text="§1º First paragraph.\n\n§2º Second paragraph.",
        source_file=Path("doc.pdf"),
    )
    segments = segment_page(page)
    assert len(segments) == 2
    assert "§1º" in segments[0].article_numbers
    assert "§2º" in segments[1].article_numbers


def test_segment_page_no_markers() -> None:
    """Page without structural markers produces 1 segment."""
    page = Page(
        document_id="doc",
        page_number=1,
        text="Plain text without markers.\n\nMore content.",
        source_file=Path("doc.pdf"),
    )
    segments = segment_page(page)
    assert len(segments) == 1
    assert segments[0].text == "Plain text without markers.\n\nMore content."
    assert segments[0].segment_index == 0


def test_segment_page_metadata_preserved() -> None:
    """Segments inherit document_id, page_number, section_title."""
    page = Page(
        document_id="x_manual",
        page_number=5,
        text="Art. 1º Content.",
        source_file=Path("doc.pdf"),
        section_title="1 Chaves Pix",
    )
    segments = segment_page(page)
    assert len(segments) == 1
    assert segments[0].document_id == "x_manual"
    assert segments[0].page_number == 5
    assert segments[0].section_title == "1 Chaves Pix"
    assert segments[0].source_file == "doc.pdf"


def test_segment_text_integrity() -> None:
    """Segmentation does not lose text: sum of segment texts equals page text."""
    page = Page(
        document_id="doc",
        page_number=1,
        text="Intro text.\n\nArt. 1º First.\n\nArt. 2º Second.",
        source_file=Path("doc.pdf"),
    )
    segments = segment_page(page)
    reconstructed = "\n\n".join(s.text for s in segments)
    assert reconstructed == page.text


def test_segment_char_offsets_non_overlapping() -> None:
    """Segment char_start/char_end do not overlap."""
    page = Page(
        document_id="doc",
        page_number=1,
        text="Intro\n\nArt. 1º First\n\nArt. 2º Second",
        source_file=Path("doc.pdf"),
    )
    segments = segment_page(page)
    for i, seg in enumerate(segments):
        if seg.char_start is not None and seg.char_end is not None:
            assert seg.char_end > seg.char_start
            if i > 0 and segments[i - 1].char_end is not None:
                assert seg.char_start >= segments[i - 1].char_end


def test_is_structural_marker_article() -> None:
    """_is_structural_marker detects Art. Xº."""
    assert _is_structural_marker("Art. 1º Content") is True
    assert _is_structural_marker("Art. 2º") is True


def test_is_structural_marker_paragraph() -> None:
    """_is_structural_marker detects §Xº."""
    assert _is_structural_marker("§1º Content") is True
    assert _is_structural_marker("§2º") is True


def test_is_structural_marker_section() -> None:
    """_is_structural_marker detects numbered subsections."""
    assert _is_structural_marker("3.1 Fluxo de registro") is True
    assert _is_structural_marker("4.2.1 Subsection") is True


def test_is_structural_marker_false() -> None:
    """_is_structural_marker rejects non-markers."""
    assert _is_structural_marker("Plain text") is False
    assert _is_structural_marker("") is False


def test_segment_page_empty() -> None:
    """Empty page produces single empty segment."""
    page = Page(
        document_id="doc",
        page_number=1,
        text="",
        source_file=Path("doc.pdf"),
    )
    segments = segment_page(page)
    assert len(segments) == 1
    assert segments[0].text == ""
    assert segments[0].segment_index == 0


def test_segment_page_whitespace_only_blocks_ignored() -> None:
    """Whitespace-only blocks are filtered out."""
    page = Page(
        document_id="doc",
        page_number=1,
        text="Content\n\n\n\nMore",
        source_file=Path("doc.pdf"),
    )
    segments = segment_page(page)
    assert len(segments) >= 1
    assert "Content" in segments[0].text


def test_segment_document() -> None:
    """segment_document processes all pages."""
    doc = Document(
        source_file=Path("doc.pdf"),
        pages=[
            Page(
                document_id="doc",
                page_number=1,
                text="Art. 1º Page 1",
                source_file=Path("doc.pdf"),
            ),
            Page(
                document_id="doc",
                page_number=2,
                text="Art. 2º Page 2",
                source_file=Path("doc.pdf"),
            ),
        ],
        total_pages=2,
    )
    segments = segment_document(doc)
    assert len(segments) == 2
    assert segments[0].page_number == 1
    assert segments[1].page_number == 2


def test_segment_records() -> None:
    """segment_records converts JSONL records to segment dicts."""
    records = [
        {
            "document_id": "doc",
            "page_number": 1,
            "section_title": "1 Chaves",
            "article_numbers": [],
            "source_file": "doc.pdf",
            "text": "Art. 1º A\n\nArt. 2º B",
        }
    ]
    segments = segment_records(records)
    assert len(segments) == 2
    assert segments[0]["document_id"] == "doc"
    assert segments[0]["page_number"] == 1
    assert "Art. 1º" in segments[0]["article_numbers"]
