#!/usr/bin/env python
"""
Run full RAG evaluation pipeline.

This script executes the evaluation framework for the RAG system, including:

- retrieval evaluation
- RAG groundedness evaluation (when LLM available)
- metric reporting
- Phoenix tracing (optional)

Run from project root:

    python scripts/evaluate_rag.py

Dependencies:

- Weaviate (required)
- Ollama / LLM runtime (optional for RAG evaluation)
- Phoenix (optional for tracing)

If Phoenix tracing is desired:

    python -m phoenix.server.main serve

Traces will appear at:

    http://localhost:6006
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


from src.evaluation.evaluation_runner import (
    export_report,
    run_full_evaluation,
)

from src.retrieval import retrieve
from src.utils.system_checks import (
    check_evaluation_dependencies,
    check_rag_dependencies,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "rag_evaluation_dataset.json"
REPORTS_DIR = PROJECT_ROOT / "reports"

K = 5


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
                "RAG generation unavailable, running retrieval-only evaluation: %s",
                e,
            )

    results = run_full_evaluation(
        DATASET_PATH,
        retriever_fn,
        rag_fn,
        k=K,
    )

    logger.info(
        "Retrieval metrics: Precision@%d = %.4f | Recall@%d = %.4f",
        K,
        results["retrieval"].get(f"precision@{K}", 0),
        K,
        results["retrieval"].get(f"recall@{K}", 0),
    )

    if results.get("rag"):
        logger.info(
            "RAG metrics: citation_coverage = %.4f | hallucination_rate = %.4f",
            results["rag"]["citation_coverage"],
            results["rag"]["hallucination_rate"],
        )

    REPORTS_DIR.mkdir(exist_ok=True)

    report_path = REPORTS_DIR / "evaluation_report.json"

    export_report(results, report_path)

    logger.info("Evaluation report saved to %s", report_path)


if __name__ == "__main__":
    main()