#!/usr/bin/env python
"""
Chunking pipeline: load pages → segment → chunk → save to JSONL.

Run from project root:

    python scripts/run_ingestion.py   # first, to generate corpus_pages.jsonl
    python scripts/run_chunking.py
"""

import json
import logging
import sys
from pathlib import Path

# Ensure project root is available when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.serializer import save_chunks_jsonl, validate_chunk_dataset  # noqa: E402
from src.chunking.structural_segmenter import segment_page  # noqa: E402
from src.chunking.token_chunker import chunk_segment  # noqa: E402
from src.ingestion.models import Page  # noqa: E402
from src.config.logging import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

PAGES_PATH = PROJECT_ROOT / "data" / "processed" / "corpus_pages.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "corpus_chunks.jsonl"


def _iter_page_records(path: Path):
    """Yield page records from JSONL file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    """Run full chunking pipeline: load pages → segment → chunk → save."""
    if not PAGES_PATH.exists():
        logger.error("Pages dataset not found: %s", PAGES_PATH)
        logger.error("Run first: python scripts/run_ingestion.py")
        raise SystemExit(1)

    logger.info("Loaded pages dataset: %s", PAGES_PATH)

    def generate_chunks():
        """Stream chunks from pages without loading all into memory."""
        for rec in _iter_page_records(PAGES_PATH):
            page = Page(
                document_id=rec["document_id"],
                page_number=rec["page_number"],
                text=rec["text"],
                source_file=Path(rec["source_file"]),
                section_title=rec.get("section_title"),
                article_numbers=rec.get("article_numbers", []),
            )
            for segment in segment_page(page):
                for chunk in chunk_segment(segment):
                    yield chunk

    records_written = save_chunks_jsonl(generate_chunks(), OUTPUT_PATH)
    logger.info("Generated %d chunks", records_written)
    logger.info("Saved dataset to %s", OUTPUT_PATH)

    # Validate dataset
    count, errors = validate_chunk_dataset(OUTPUT_PATH)
    if errors:
        for err in errors[:5]:
            logger.warning("Validation: %s", err)
        if len(errors) > 5:
            logger.warning("... and %d more validation errors", len(errors) - 5)
    else:
        logger.info("Dataset validation passed (%d records)", count)


if __name__ == "__main__":
    main()
