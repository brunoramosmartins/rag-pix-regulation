#!/usr/bin/env python
"""
Evaluate retrieval system against evaluation dataset.

Run from project root (after run_indexing.py):

    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --strategy hybrid
    python scripts/evaluate_retrieval.py --strategy vector
    python scripts/evaluate_retrieval.py --strategy keyword
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluation_runner import run_retrieval_evaluation
from src.retrieval import retrieve
from src.utils.system_checks import check_evaluation_dependencies
from src.config.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "rag_evaluation_dataset.json"
DEFAULT_K = 5


def main() -> None:
    """Run retrieval evaluation and display metrics."""
    parser = argparse.ArgumentParser(description="Evaluate retrieval system")
    parser.add_argument(
        "--strategy",
        choices=["vector", "keyword", "hybrid"],
        default=None,
        help="Search strategy override (default: use config.yaml)",
    )
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        logger.error("Dataset not found: %s", DATASET_PATH)
        raise SystemExit(1)

    ready, msg = check_evaluation_dependencies()
    if not ready:
        logger.error("%s", msg)
        raise SystemExit(1)

    strategy_label = args.strategy or "config default"

    def retriever_fn(query: str):
        return retrieve(query, top_k=DEFAULT_K, search_strategy=args.strategy)

    metrics = run_retrieval_evaluation(DATASET_PATH, retriever_fn, k=DEFAULT_K)

    logger.info("Retrieval evaluation (k=%d, strategy=%s)", DEFAULT_K, strategy_label)
    logger.info("Queries evaluated: %d", metrics.get("n_queries", 0))
    logger.info(
        "Precision@%d = %.4f", DEFAULT_K, metrics.get(f"precision@{DEFAULT_K}", 0)
    )
    logger.info("Recall@%d = %.4f", DEFAULT_K, metrics.get(f"recall@{DEFAULT_K}", 0))
    logger.info("NDCG@%d = %.4f", DEFAULT_K, metrics.get(f"ndcg@{DEFAULT_K}", 0))
    logger.info("MAP@%d = %.4f", DEFAULT_K, metrics.get(f"map@{DEFAULT_K}", 0))

    if metrics.get("n_queries", 0) == 0:
        logger.warning(
            "No queries with ground truth. Add expected_pages to %s", DATASET_PATH
        )


if __name__ == "__main__":
    main()
