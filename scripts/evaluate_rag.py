#!/usr/bin/env python
"""
Run full RAG evaluation pipeline with answer quality metrics.

Evaluates retrieval, groundedness, hallucination, and answer quality
(semantic similarity + concept coverage) against expected answers.

Run from project root:

    python scripts/evaluate_rag.py
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Optional Phoenix tracing
try:
    from phoenix.otel import register

    register(project_name="rag-pix-regulation", auto_instrument=False)
except ImportError:
    pass

from src.evaluation.evaluation_runner import export_report, run_full_evaluation  # noqa: E402
from src.retrieval import retrieve  # noqa: E402
from src.utils.system_checks import (  # noqa: E402
    check_evaluation_dependencies,
    check_rag_dependencies,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "rag_evaluation_dataset.json"
REPORTS_DIR = PROJECT_ROOT / "reports"

K = 5


def _print_results(results: dict) -> None:
    """Print formatted evaluation results table."""
    ret = results.get("retrieval", {})
    rag = results.get("rag")
    by_diff = results.get("by_difficulty", {})

    print("\n" + "=" * 70)
    print(f"  RAG Evaluation Results (k={K})")
    print("=" * 70)

    # Retrieval metrics
    print("\n  Retrieval:")
    print(
        f"    P@{K}={ret.get(f'precision@{K}', 0):.4f}  "
        f"R@{K}={ret.get(f'recall@{K}', 0):.4f}  "
        f"NDCG@{K}={ret.get(f'ndcg@{K}', 0):.4f}  "
        f"MAP@{K}={ret.get(f'map@{K}', 0):.4f}"
    )

    if rag:
        # Answer quality
        print("\n  Answer Quality:")
        print(
            f"    Similarity={rag.get('answer_similarity_avg', 0):.4f}  "
            f"Concept Coverage={rag.get('concept_coverage_avg', 0):.4f}  "
            f"Quality Score={rag.get('quality_score_avg', 0):.4f}"
        )

        # Groundedness
        print("\n  Groundedness:")
        print(
            f"    Citation Coverage={rag.get('citation_coverage', 0):.4f}  "
            f"Hallucination Rate={rag.get('hallucination_rate', 0):.4f}  "
            f"Groundedness={rag.get('groundedness_avg', 0):.4f}"
        )

        # By difficulty tier
        if by_diff:
            print("\n  By Difficulty:")
            print(
                f"    {'Tier':<16} {'N':>4} {'P@5':>7} {'R@5':>7} "
                f"{'Sim':>7} {'CC':>7} {'QS':>7}"
            )
            print("    " + "-" * 55)
            for tier, m in sorted(by_diff.items()):
                print(
                    f"    {tier:<16} {m.get('n_queries', 0):>4d} "
                    f"{m.get('precision_at_k', 0):>7.4f} "
                    f"{m.get('recall_at_k', 0):>7.4f} "
                    f"{m.get('answer_similarity', 0):>7.4f} "
                    f"{m.get('concept_coverage', 0):>7.4f} "
                    f"{m.get('quality_score', 0):>7.4f}"
                )

        print(f"\n  Queries evaluated: {rag.get('n_queries', 0)}")
    else:
        print("\n  RAG evaluation skipped (LLM not available)")

    print("=" * 70)


def main() -> None:
    """Run evaluation pipeline."""
    if not DATASET_PATH.exists():
        logger.error("Dataset not found: %s", DATASET_PATH)
        sys.exit(1)

    ready, msg = check_evaluation_dependencies()
    if not ready:
        logger.error("%s", msg)
        sys.exit(1)

    def retriever_fn(query: str):
        return retrieve(query, top_k=K)

    rag_fn = None

    rag_ready, rag_msg = check_rag_dependencies()

    if rag_ready:
        try:
            from src.llm import BaselineLLM
            from src.rag.rag_pipeline import answer_query

            llm = BaselineLLM()

            def _rag_fn(q: str):
                return answer_query(q, llm=llm, top_k=K)

            rag_fn = _rag_fn

        except ImportError as e:
            logger.warning(
                "RAG not available, running retrieval-only evaluation: %s",
                e,
            )

    results = run_full_evaluation(
        DATASET_PATH,
        retriever_fn,
        rag_fn,
        k=K,
    )

    _print_results(results)

    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / "evaluation_report.json"
    export_report(results, report_path)


if __name__ == "__main__":
    main()
