#!/usr/bin/env python
"""
Ingestion pipeline: load PDFs, extract pages, serialize to JSONL.

Run from project root:

    python scripts/run_ingestion.py
"""

import logging
import sys
from pathlib import Path

# Ensure project root is available when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import load_pdfs_from_dir  # noqa: E402
from src.ingestion.serializer import save_documents_jsonl  # noqa: E402
from src.config.logging import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    """Run full ingestion pipeline: load → serialize → save."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    output_file = PROJECT_ROOT / "data" / "processed" / "corpus_pages.jsonl"

    if not raw_dir.exists():
        logger.error("Directory not found: %s", raw_dir)
        logger.error("Place regulatory PDFs inside data/raw/")
        raise SystemExit(1)

    # 1. Load documents
    logger.info("Loading documents from %s", raw_dir)
    documents = load_pdfs_from_dir(raw_dir)

    if not documents:
        logger.warning("No PDF files found in %s", raw_dir)
        return

    # 2. Extract pages (already done in load_pdfs_from_dir)
    total_pages = sum(doc.total_pages for doc in documents)
    logger.info("Parsed %d document(s), %d pages total", len(documents), total_pages)

    # 3 & 4. Serialize and write dataset
    records_written = save_documents_jsonl(documents, output_file, skip_invalid=True)
    logger.info("Generated %d records", records_written)
    logger.info("Saved dataset to %s", output_file)


if __name__ == "__main__":
    main()
