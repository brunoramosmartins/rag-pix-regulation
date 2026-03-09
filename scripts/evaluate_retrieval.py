#!/usr/bin/env python
"""
Evaluate retrieval system against evaluation dataset.

Run from project root (after run_indexing.py):

    python scripts/evaluate_retrieval.py
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluation_runner import run_retrieval_evaluation
from src.retrieval import retrieve
from src.utils.system_checks import check_evaluation_dependencies

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "rag_evaluation_dataset.json"
DEFAULT_K = 5


def main() -> None:
    """Run retrieval evaluation and display metrics."""
    if not DATASET_PATH.exists():
        logger.error("Dataset not found: %s", DATASET_PATH)
        raise SystemExit(1)

    ready, msg = check_evaluation_dependencies()
    if not ready:
        logger.error("%s", msg)
        raise SystemExit(1)

    def retriever_fn(query: str):
        return retrieve(query, top_k=DEFAULT_K)

    metrics = run_retrieval_evaluation(DATASET_PATH, retriever_fn, k=DEFAULT_K)

    logger.info("Retrieval evaluation (k=%d)", DEFAULT_K)
    logger.info("Queries evaluated: %d", metrics.get("n_queries", 0))
    logger.info("Precision@%d = %.4f", DEFAULT_K, metrics.get(f"precision@{DEFAULT_K}", 0))
    logger.info("Recall@%d = %.4f", DEFAULT_K, metrics.get(f"recall@{DEFAULT_K}", 0))

    if metrics.get("n_queries", 0) == 0:
        logger.warning("No queries with ground truth. Add expected_pages to %s", DATASET_PATH)


if __name__ == "__main__":
    main()
