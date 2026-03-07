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

__all__ = [
    "Document",
    "Page",
    "load_pdf",
    "load_pdfs_from_dir",
]