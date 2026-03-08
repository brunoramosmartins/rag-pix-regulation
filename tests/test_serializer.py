"""Unit tests for document serialization."""

import json
from pathlib import Path

from src.ingestion.models import Document, Page
from src.ingestion.serializer import (
    document_to_records,
    generate_document_id,
    page_to_record,
    save_documents_jsonl,
    save_records_jsonl,
)


def test_generate_document_id() -> None:
    """generate_document_id normalizes filename to lowercase with underscores."""
    assert generate_document_id(Path("X_ManualOperacionaldoDICT.pdf")) == "x_manualoperacionaldodict"
    assert generate_document_id(Path("Doc With Spaces.pdf")) == "doc_with_spaces"


def test_generate_document_id_stem_only() -> None:
    """generate_document_id uses stem (no extension)."""
    assert generate_document_id(Path("/path/to/file.pdf")) == "file"


def test_page_to_record_flat_schema() -> None:
    """page_to_record produces flat schema with all required fields."""
    page = Page(
        document_id="doc",
        page_number=5,
        text="Page content",
        source_file=Path("X_ManualOperacionaldoDICT.pdf"),
        section_title="1 Chaves Pix",
        article_numbers=["Art. 1º"],
    )
    record = page_to_record(page)

    assert record["document_id"] == "x_manualoperacionaldodict"
    assert record["page_number"] == 5
    assert record["section_title"] == "1 Chaves Pix"
    assert record["article_numbers"] == ["Art. 1º"]
    assert record["source_file"] == "X_ManualOperacionaldoDICT.pdf"
    assert record["text"] == "Page content"


def test_document_to_records() -> None:
    """document_to_records converts all pages to records."""
    doc = Document(
        source_file=Path("test.pdf"),
        pages=[
            Page(
                document_id="test",
                page_number=1,
                text="Valid text content.",
                source_file=Path("test.pdf"),
            ),
            Page(
                document_id="test",
                page_number=2,
                text="More content.",
                source_file=Path("test.pdf"),
            ),
        ],
        total_pages=2,
    )
    records = document_to_records(doc)
    assert len(records) == 2
    assert records[0]["page_number"] == 1
    assert records[1]["page_number"] == 2


def test_document_to_records_skips_invalid() -> None:
    """document_to_records skips pages with invalid text when skip_invalid=True."""
    doc = Document(
        source_file=Path("test.pdf"),
        pages=[
            Page(
                document_id="test",
                page_number=1,
                text="",  # Invalid: empty
                source_file=Path("test.pdf"),
            ),
            Page(
                document_id="test",
                page_number=2,
                text="Valid content.",
                source_file=Path("test.pdf"),
            ),
        ],
        total_pages=2,
    )
    records = document_to_records(doc, skip_invalid=True)
    assert len(records) == 1
    assert records[0]["page_number"] == 2


def test_save_records_jsonl(tmp_path: Path) -> None:
    """save_records_jsonl writes valid JSONL incrementally."""
    output = tmp_path / "out.jsonl"
    records = [
        {"document_id": "doc1", "page_number": 1, "text": "A"},
        {"document_id": "doc1", "page_number": 2, "text": "B"},
    ]
    save_records_jsonl(records, output)

    lines = output.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0]) == records[0]
    assert json.loads(lines[1]) == records[1]


def test_save_documents_jsonl(tmp_path: Path) -> None:
    """save_documents_jsonl writes documents and returns record count."""
    output = tmp_path / "corpus.jsonl"
    doc = Document(
        source_file=Path("test.pdf"),
        pages=[
            Page(
                document_id="test",
                page_number=1,
                text="Content.",
                source_file=Path("test.pdf"),
            ),
        ],
        total_pages=1,
    )
    count = save_documents_jsonl([doc], output)
    assert count == 1
    assert output.exists()
    record = json.loads(output.read_text(encoding="utf-8").strip())
    assert record["document_id"] == "test"
    assert record["text"] == "Content."
