"""Unit tests for PDF ingestion utilities."""

from pathlib import Path

import pytest

from src.ingestion import load_pdf, load_pdfs_from_dir
from src.ingestion.models import Document, Page


def test_load_pdf_file_not_found() -> None:
    """load_pdf should raise FileNotFoundError when the file does not exist."""
    missing_file = Path("nonexistent.pdf")

    with pytest.raises(FileNotFoundError, match="PDF not found"):
        load_pdf(missing_file)


def test_load_pdf_invalid_extension(tmp_path: Path) -> None:
    """load_pdf should raise ValueError when the file is not a PDF."""
    txt_file = tmp_path / "file.txt"
    txt_file.write_text("not a pdf")

    with pytest.raises(ValueError, match="Expected a PDF file"):
        load_pdf(txt_file)


def test_load_pdfs_from_dir_empty(tmp_path: Path) -> None:
    """load_pdfs_from_dir should return an empty list when no PDFs exist."""
    result = load_pdfs_from_dir(tmp_path)

    assert result == []


def test_load_pdfs_from_dir_not_a_directory(tmp_path: Path) -> None:
    """load_pdfs_from_dir should raise NotADirectoryError when path is not a directory."""
    fake_file = tmp_path / "file.pdf"
    fake_file.write_text("not really a pdf")

    with pytest.raises(NotADirectoryError, match="Not a directory"):
        load_pdfs_from_dir(fake_file)


def test_document_model_full_text() -> None:
    """Document.full_text should concatenate page texts in order."""
    doc = Document(
        source_file=Path("test.pdf"),
        pages=[
            Page(
                document_id="test",
                page_number=1,
                text="Page 1",
                source_file=Path("test.pdf"),
            ),
            Page(
                document_id="test",
                page_number=2,
                text="Page 2",
                source_file=Path("test.pdf"),
            ),
        ],
        total_pages=2,
    )

    assert doc.full_text == "Page 1\n\nPage 2"


def test_page_document_id() -> None:
    """Page stores document_id for tracing and citation."""
    page = Page(
        document_id="X_ManualOperacionaldoDICT",
        page_number=1,
        text="text",
        source_file=Path("X_ManualOperacionaldoDICT.pdf"),
    )
    assert page.document_id == "X_ManualOperacionaldoDICT"