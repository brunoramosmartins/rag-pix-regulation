#!/usr/bin/env python
"""
Initialize Weaviate collection schema for chunk indexing.

Run from project root:

    python scripts/init_weaviate.py
    python scripts/init_weaviate.py --recreate   # Force schema recreation (deletes existing data)
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vectorstore.weaviate_client import (  # noqa: E402
    init_chunk_collection,
    is_weaviate_ready,
    validate_chunk_schema,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Initialize Weaviate Chunk collection."""
    parser = argparse.ArgumentParser(
        description="Initialize Weaviate collection schema"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collection (deletes existing indexed data)",
    )
    args = parser.parse_args()

    if not is_weaviate_ready():
        logger.error("Weaviate is not running. Start Weaviate first (e.g. via Docker).")
        raise SystemExit(1)

    init_chunk_collection(recreate=args.recreate)
    validate_chunk_schema()
    logger.info("Weaviate collection initialized successfully")


if __name__ == "__main__":
    main()
