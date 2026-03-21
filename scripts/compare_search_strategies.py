#!/usr/bin/env python
"""
Compare retrieval strategies: vector, keyword, and hybrid (with alpha sweep).

Run from project root (after run_indexing.py):

    python scripts/compare_search_strategies.py
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluation_runner import run_retrieval_evaluation  # noqa: E402
from src.retrieval import retrieve  # noqa: E402
from src.utils.system_checks import check_evaluation_dependencies  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "rag_evaluation_dataset.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_K = 5

# Strategies to compare: (label, strategy, alpha)
STRATEGIES = [
    ("vector", "vector", None),
    ("keyword (BM25)", "keyword", None),
    ("hybrid α=0.25", "hybrid", 0.25),
    ("hybrid α=0.50", "hybrid", 0.50),
    ("hybrid α=0.75", "hybrid", 0.75),
]


def main() -> None:
    """Run evaluation for each strategy and produce comparison table."""
    if not DATASET_PATH.exists():
        logger.error("Dataset not found: %s", DATASET_PATH)
        raise SystemExit(1)

    ready, msg = check_evaluation_dependencies()
    if not ready:
        logger.error("%s", msg)
        raise SystemExit(1)

    all_results: dict[str, dict] = {}

    for label, strategy, alpha in STRATEGIES:
        logger.info("=" * 60)
        logger.info("Evaluating: %s", label)
        logger.info("=" * 60)

        def retriever_fn(query: str, _s=strategy, _a=alpha):
            return retrieve(query, top_k=DEFAULT_K, search_strategy=_s, alpha=_a)

        metrics = run_retrieval_evaluation(DATASET_PATH, retriever_fn, k=DEFAULT_K)
        all_results[label] = metrics

        logger.info(
            "%s → P@%d=%.4f  R@%d=%.4f  NDCG@%d=%.4f  MAP@%d=%.4f  (n=%d)",
            label,
            DEFAULT_K, metrics.get(f"precision@{DEFAULT_K}", 0),
            DEFAULT_K, metrics.get(f"recall@{DEFAULT_K}", 0),
            DEFAULT_K, metrics.get(f"ndcg@{DEFAULT_K}", 0),
            DEFAULT_K, metrics.get(f"map@{DEFAULT_K}", 0),
            metrics.get("n_queries", 0),
        )

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"{'Strategy':<20} {'P@5':>8} {'R@5':>8} {'NDCG@5':>8} {'MAP@5':>8} {'Queries':>8}")
    print("-" * 80)
    for label in all_results:
        m = all_results[label]
        print(
            f"{label:<20} {m.get(f'precision@{DEFAULT_K}', 0):>8.4f} "
            f"{m.get(f'recall@{DEFAULT_K}', 0):>8.4f} "
            f"{m.get(f'ndcg@{DEFAULT_K}', 0):>8.4f} "
            f"{m.get(f'map@{DEFAULT_K}', 0):>8.4f} "
            f"{m.get('n_queries', 0):>8d}"
        )
    print("=" * 80)

    # Export report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "strategy_comparison.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
