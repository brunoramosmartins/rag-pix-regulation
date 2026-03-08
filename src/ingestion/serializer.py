"""Serialization of parsed documents to structured JSONL format."""

import json
from pathlib import Path
from typing import Iterable

from .models import Document, Page
from .text_cleaner import validate_text


def generate_document_id(source_file: Path) -> str:
    """
    Generate a stable, normalized document identifier.

    Deterministic, filesystem-independent, safe for vector database IDs.
    """
    return source_file.stem.lower().replace(" ", "_")


def page_to_record(page: Page) -> dict:
    """
    Convert a Page to a flat JSONL record.

    Flat schema improves vector DB ingestion and retrieval filtering.
    """
    return {
        "document_id": generate_document_id(page.source_file),
        "page_number": page.page_number,
        "section_title": page.section_title,
        "article_numbers": page.article_numbers,
        "source_file": page.source_file.name,
        "text": page.text,
    }


def document_to_records(
    document: Document,
    *,
    skip_invalid: bool = True,
) -> list[dict]:
    """
    Convert a Document to a list of JSONL records.

    Skips pages with invalid text when skip_invalid is True.
    """
    records: list[dict] = []
    for page in document.pages:
        if skip_invalid and not validate_text(page.text):
            continue
        records.append(page_to_record(page))
    return records


def save_records_jsonl(records: Iterable[dict], path: Path) -> None:
    """
    Write records to JSONL file incrementally.

    Streaming write supports large corpora without loading all in memory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def save_documents_jsonl(
    documents: list[Document],
    path: Path,
    *,
    skip_invalid: bool = True,
) -> int:
    """
    Serialize documents to JSONL and write to file.

    Returns the number of records written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for document in documents:
            for page in document.pages:
                if skip_invalid and not validate_text(page.text):
                    continue
                record = page_to_record(page)
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
                count += 1

    return count
