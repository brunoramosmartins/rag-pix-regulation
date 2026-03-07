"""PDF ingestion utilities using PyMuPDF."""

from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from .models import Document, Page


def load_pdf(path: Path) -> Document:
    """
    Extract text from a PDF file while preserving page boundaries.

    The function reads the PDF page-by-page and converts each page
    into a `Page` model instance. The resulting `Document` preserves
    page-level segmentation, which is required for downstream
    chunking, retrieval attribution, and citation generation.

    Parameters
    ----------
    path : Path
        Path to the PDF document.

    Returns
    -------
    Document
        Parsed document containing page-level text.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.

    ValueError
        If the file is not a PDF.

    RuntimeError
        If the PDF cannot be opened or parsed.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    pages: List[Page] = []

    try:
        with fitz.open(path) as pdf:
            for page_index in range(pdf.page_count):
                page = pdf.load_page(page_index)

                # Extract text preserving layout structure
                text = page.get_text("text", sort=True)

                # Normalize encoding and whitespace
                text = text.encode("utf-8", errors="replace").decode("utf-8").strip()

                pages.append(
                    Page(
                        page_number=page_index + 1,
                        text=text,
                        source_file=path,
                    )
                )

    except Exception as exc:
        raise RuntimeError(f"Failed to parse PDF: {path}") from exc

    return Document(
        source_file=path,
        pages=pages,
        total_pages=len(pages),
    )


def load_pdfs_from_dir(directory: Path) -> List[Document]:
    """
    Load and parse all PDF documents in a directory.

    Parameters
    ----------
    directory : Path
        Directory containing PDF files.

    Returns
    -------
    List[Document]
        List of parsed documents.

    Raises
    ------
    NotADirectoryError
        If the provided path is not a directory.
    """
    directory = Path(directory)

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    documents: List[Document] = []

    for pdf_path in sorted(directory.glob("*.pdf")):
        documents.append(load_pdf(pdf_path))

    return documents