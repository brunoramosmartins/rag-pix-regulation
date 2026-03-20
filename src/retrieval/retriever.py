"""Retriever orchestration: query embedding + vector search + optional reranking."""

import logging
from pathlib import Path

from .models import RetrievalResult
from .query_embedding import embed_query
from .vector_search import vector_search

logger = logging.getLogger(__name__)


def _load_reranking_config() -> dict:
    """Load reranking config from config.yaml. Returns empty dict on failure."""
    try:
        import yaml

        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            return cfg.get("reranking", {})
    except Exception:
        pass
    return {}


def retrieve(
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.0,
) -> list[RetrievalResult]:
    """
    Retrieve top-K most relevant chunks for a query.

    Pipeline: query → embed → vector_search → [optional rerank] → results.

    When reranking is enabled via config, over-retrieves (top_k * 2) from
    vector search then reranks down to top_k.

    Returns
    -------
    list[RetrievalResult]
        Ranked retrieval results ordered by similarity/relevance score.
    """
    rerank_cfg = _load_reranking_config()
    reranking_enabled = rerank_cfg.get("enabled", False)

    # Over-retrieve when reranking to give cross-encoder more candidates
    fetch_k = top_k * 2 if reranking_enabled else top_k

    query_vector = embed_query(query)
    raw_results = vector_search(
        query_vector, top_k=fetch_k, min_similarity=min_similarity,
    )

    results = [
        RetrievalResult(
            text=r.get("text") or "",
            chunk_id=r.get("chunk_id") or "",
            document_id=r.get("document_id") or "",
            page_number=r.get("page_number") or 0,
            section_title=r.get("section_title"),
            similarity_score=r.get("similarity_score"),
            source_file=r.get("source_file"),
        )
        for r in raw_results
    ]

    if reranking_enabled and results:
        from .reranker import rerank

        model_name = rerank_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        top_n = rerank_cfg.get("top_n", top_k)
        logger.info(
            "Reranking %d candidates → top %d with %s",
            len(results), top_n, model_name,
        )
        results = rerank(query, results, model_name=model_name, top_n=top_n)

    return results
