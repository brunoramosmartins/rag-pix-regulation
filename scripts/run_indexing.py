#!/usr/bin/env python
"""
Indexing pipeline: load chunks → generate embeddings → index into Weaviate.

Run from project root:

    python scripts/run_indexing.py
    python scripts/run_indexing.py --batch-size 64
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vectorstore.indexer import index_chunks  # noqa: E402
from src.vectorstore.weaviate_client import is_weaviate_ready  # noqa: E402
from src.config.logging import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "corpus_chunks.jsonl"


def main() -> None:
    """Run full indexing pipeline."""
    parser = argparse.ArgumentParser(description="Index chunks into Weaviate")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of vectors per Weaviate insert batch (default: 100)",
    )
    args = parser.parse_args()

    if not CHUNKS_PATH.exists():
        logger.error("Chunks dataset not found: %s", CHUNKS_PATH)
        logger.error("Run first: python scripts/run_chunking.py")
        raise SystemExit(1)

    if not is_weaviate_ready():
        logger.error("Weaviate is not running. Start Weaviate first (e.g. via Docker).")
        raise SystemExit(1)

    logger.info("Loading chunks from %s", CHUNKS_PATH)
    count = index_chunks(CHUNKS_PATH, batch_size=args.batch_size)
    logger.info("Indexed %d chunks successfully", count)


if __name__ == "__main__":
    main()
