#!/usr/bin/env python
"""
Utility script to manually validate the PDF ingestion pipeline.

Example
-------
Run from the project root:

    python scripts/test_pdf_loader.py
"""

from pathlib import Path
import logging
import sys

# Ensure project root is available when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import load_pdfs_from_dir  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Run ingestion pipeline against the local raw dataset."""

    raw_dir = PROJECT_ROOT / "data" / "raw"

    if not raw_dir.exists():
        logger.error("Directory not found: %s", raw_dir)
        logger.error("Place regulatory PDFs inside data/raw/")
        return

    logger.info("Loading PDFs from %s", raw_dir)

    try:
        documents = load_pdfs_from_dir(raw_dir)
    except Exception as exc:
        logger.exception("Failed to load documents")
        raise SystemExit(1) from exc

    if not documents:
        logger.warning("No PDF files found in %s", raw_dir)
        return

    logger.info("Loaded %d document(s)", len(documents))

    for doc in documents:
        logger.info("Document: %s", doc.source_file.name)
        logger.info("Total pages: %d", doc.total_pages)

        if doc.pages:
            preview = doc.pages[0].text[:500].replace("\n", " ")
            logger.info("Page 1 preview: %s...", preview)

        logger.info("-" * 40)


if __name__ == "__main__":
    main()
