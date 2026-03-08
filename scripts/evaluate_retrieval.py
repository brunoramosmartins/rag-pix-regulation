#!/usr/bin/env python
"""
Evaluate retrieval system against evaluation dataset.

Run from project root (after run_indexing.py and populating retrieval_dataset.json):

    python scripts/evaluate_retrieval.py
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import evaluate_retrieval  # noqa: E402
from src.retrieval import retrieve  # noqa: E402
from src.vectorstore.weaviate_client import is_weaviate_ready  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_dataset.json"
DEFAULT_K = 5


def main() -> None:
    """Run retrieval evaluation and display metrics."""
    if not DATASET_PATH.exists():
        logger.error("Dataset not found: %s", DATASET_PATH)
        raise SystemExit(1)

    if not is_weaviate_ready():
        logger.error("Weaviate is not running. Run: python scripts/run_indexing.py")
        raise SystemExit(1)

    def retriever_fn(query: str):
        return retrieve(query, top_k=DEFAULT_K)

    metrics = evaluate_retrieval(DATASET_PATH, retriever_fn, k=DEFAULT_K)

    logger.info("Retrieval evaluation (k=%d)", DEFAULT_K)
    logger.info("Queries evaluated: %d", metrics.get("n_queries", 0))
    logger.info(
        "Precision@%d = %.4f",
        DEFAULT_K,
        metrics.get(f"precision@{DEFAULT_K}", 0),
    )
    logger.info(
        "Recall@%d = %.4f",
        DEFAULT_K,
        metrics.get(f"recall@{DEFAULT_K}", 0),
    )

    if metrics.get("n_queries", 0) == 0:
        logger.warning(
            "No queries with ground truth. Add relevant_chunk_ids to %s",
            DATASET_PATH,
        )


if __name__ == "__main__":
    main()
