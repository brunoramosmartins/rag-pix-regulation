"""
Ingestion module.

This package contains utilities for loading and preprocessing regulatory
documents prior to the chunking and embedding stages of the RAG pipeline.

Public API
----------
Document
    Structured representation of a parsed document.

Page
    Representation of an individual page extracted from a document.

load_pdf
    Load and parse a single PDF document.

load_pdfs_from_dir
    Load and parse all PDF files from a directory.
"""

from .models import Document, Page
from .pdf_loader import load_pdf, load_pdfs_from_dir
from .serializer import (
    document_to_records,
    generate_document_id,
    page_to_record,
    save_documents_jsonl,
    save_records_jsonl,
)

__all__ = [
    "Document",
    "Page",
    "load_pdf",
    "load_pdfs_from_dir",
    "generate_document_id",
    "page_to_record",
    "document_to_records",
    "save_records_jsonl",
    "save_documents_jsonl",
]