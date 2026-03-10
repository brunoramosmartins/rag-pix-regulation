#!/usr/bin/env python
"""
Run full RAG corpus pipeline: ingestion → chunking → Weaviate init → indexing.

Run from project root:

    python scripts/run_pipeline.py
"""

import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

STEPS = [
    ("run_ingestion.py", "PDF → corpus_pages.jsonl"),
    ("run_chunking.py", "pages → corpus_chunks.jsonl"),
    ("init_weaviate.py", "Create Weaviate collection schema"),
    ("run_indexing.py", "Chunks → embeddings → Weaviate"),
]


def main() -> None:
    """Execute pipeline steps in sequence."""
    for script, desc in STEPS:
        script_path = PROJECT_ROOT / "scripts" / script
        if not script_path.exists():
            logger.error("Script not found: %s", script_path)
            raise SystemExit(1)

        logger.info("--- %s: %s ---", script, desc)
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            logger.error(
                "Pipeline failed at %s (exit code %d)", script, result.returncode
            )
            raise SystemExit(result.returncode)

    logger.info("Pipeline completed successfully")
    logger.info("Run 'python scripts/demo_retrieval.py' to validate semantic search")


if __name__ == "__main__":
    main()
