"""Retriever orchestration: query embedding + search strategy dispatch + optional reranking."""

import logging
import time
from pathlib import Path

from .models import RetrievalResult

try:
    from src.observability.tracing import span_set_input, span_set_output, trace_span
except ImportError:

    def trace_span(name: str, attributes=None, openinference_span_kind=None):
        from contextlib import nullcontext

        return nullcontext()

    def span_set_input(span, value): ...
    def span_set_output(span, value): ...


logger = logging.getLogger(__name__)

VALID_STRATEGIES = ("vector", "keyword", "hybrid")


def _load_config() -> dict:
    """Load full config from config.yaml. Returns empty dict on failure."""
    try:
        import yaml

        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def _raw_to_results(raw_results: list[dict]) -> list[RetrievalResult]:
    """Convert raw search dicts to RetrievalResult dataclass instances."""
    return [
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


def retrieve(
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.0,
    search_strategy: str | None = None,
    alpha: float | None = None,
) -> list[RetrievalResult]:
    """
    Retrieve top-K most relevant chunks for a query.

    Supports three search strategies:
    - "vector": semantic search via BGE-M3 embeddings (default)
    - "keyword": BM25 keyword search
    - "hybrid": combined BM25 + vector search

    When reranking is enabled via config, over-retrieves (top_k * 2) then
    reranks down to top_n.

    Each substep (embedding, search, reranking) is instrumented with
    OpenTelemetry spans for fine-grained observability in Phoenix.

    Parameters
    ----------
    query : str
        The user query.
    top_k : int
        Number of results to return.
    min_similarity : float
        Minimum similarity threshold (applied only for "vector" strategy).
    search_strategy : str | None
        Override search strategy. If None, reads from config.yaml
        (retrieval.search_strategy). Defaults to "vector" if not configured.
    alpha : float | None
        Override hybrid alpha. If None, reads from config.yaml
        (retrieval.hybrid.alpha). Defaults to 0.5.

    Returns
    -------
    list[RetrievalResult]
        Ranked retrieval results ordered by similarity/relevance score.
    """
    cfg = _load_config()
    retrieval_cfg = cfg.get("retrieval", {})
    rerank_cfg = cfg.get("reranking", {})
    reranking_enabled = rerank_cfg.get("enabled", False)

    # Resolve search strategy
    strategy = search_strategy or retrieval_cfg.get("search_strategy", "vector")
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Invalid search_strategy: {strategy!r}. "
            f"Must be one of: {VALID_STRATEGIES}"
        )

    # Over-retrieve when reranking to give cross-encoder more candidates
    fetch_k = top_k * 2 if reranking_enabled else top_k

    # --- Dispatch to appropriate search function with granular tracing ---

    if strategy == "keyword":
        from .keyword_search import keyword_search

        with trace_span(
            "keyword_search",
            attributes={
                "retrieval.strategy": "keyword",
                "retrieval.top_k": fetch_k,
            },
        ) as span:
            t0 = time.perf_counter()
            raw_results = keyword_search(query, top_k=fetch_k)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            results = _raw_to_results(raw_results)

            if span and span.is_recording():
                span_set_input(span, query)
                span.set_attribute("retrieval.result_count", len(results))
                span.set_attribute("retrieval.latency_ms", elapsed_ms)
                span_set_output(span, {"count": len(results), "latency_ms": elapsed_ms})

    elif strategy == "hybrid":
        from .hybrid_search import hybrid_search
        from .query_embedding import embed_query

        hybrid_cfg = retrieval_cfg.get("hybrid", {})
        resolved_alpha = alpha if alpha is not None else hybrid_cfg.get("alpha", 0.5)
        fusion_type = hybrid_cfg.get("fusion_type", "ranked")

        # Trace query embedding
        with trace_span(
            "query_embedding",
            attributes={"embedding.model": "BAAI/bge-m3"},
        ) as span:
            t0 = time.perf_counter()
            query_vector = embed_query(query)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

            if span and span.is_recording():
                span_set_input(span, query)
                span.set_attribute("embedding.dimensions", len(query_vector))
                span.set_attribute("embedding.latency_ms", elapsed_ms)
                span_set_output(span, {"dimensions": len(query_vector), "latency_ms": elapsed_ms})

        # Trace hybrid search
        with trace_span(
            "hybrid_search",
            attributes={
                "retrieval.strategy": "hybrid",
                "retrieval.top_k": fetch_k,
                "retrieval.hybrid.alpha": resolved_alpha,
                "retrieval.hybrid.fusion_type": fusion_type,
            },
        ) as span:
            t0 = time.perf_counter()
            raw_results = hybrid_search(
                query=query,
                query_vector=query_vector,
                top_k=fetch_k,
                alpha=resolved_alpha,
                fusion_type=fusion_type,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            results = _raw_to_results(raw_results)

            if span and span.is_recording():
                span_set_input(span, query)
                span.set_attribute("retrieval.result_count", len(results))
                span.set_attribute("retrieval.latency_ms", elapsed_ms)
                _set_result_attributes(span, results)
                span_set_output(span, {"count": len(results), "latency_ms": elapsed_ms})

    else:  # "vector" (default)
        from .query_embedding import embed_query
        from .vector_search import vector_search

        # Trace query embedding
        with trace_span(
            "query_embedding",
            attributes={"embedding.model": "BAAI/bge-m3"},
        ) as span:
            t0 = time.perf_counter()
            query_vector = embed_query(query)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

            if span and span.is_recording():
                span_set_input(span, query)
                span.set_attribute("embedding.dimensions", len(query_vector))
                span.set_attribute("embedding.latency_ms", elapsed_ms)
                span_set_output(span, {"dimensions": len(query_vector), "latency_ms": elapsed_ms})

        # Trace vector search
        with trace_span(
            "vector_search",
            attributes={
                "retrieval.strategy": "vector",
                "retrieval.top_k": fetch_k,
                "retrieval.min_similarity": min_similarity,
            },
        ) as span:
            t0 = time.perf_counter()
            raw_results = vector_search(
                query_vector, top_k=fetch_k, min_similarity=min_similarity,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            results = _raw_to_results(raw_results)

            if span and span.is_recording():
                span_set_input(span, query)
                span.set_attribute("retrieval.result_count", len(results))
                span.set_attribute("retrieval.latency_ms", elapsed_ms)
                _set_result_attributes(span, results)
                span_set_output(span, {"count": len(results), "latency_ms": elapsed_ms})

    # Apply reranking if enabled
    if reranking_enabled and results:
        from .reranker import rerank

        model_name = rerank_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        top_n = rerank_cfg.get("top_n", top_k)

        with trace_span(
            "reranking",
            attributes={
                "reranking.model": model_name,
                "reranking.input_count": len(results),
                "reranking.top_n": top_n,
            },
        ) as span:
            t0 = time.perf_counter()
            logger.info(
                "Reranking %d candidates → top %d with %s",
                len(results), top_n, model_name,
            )
            results = rerank(query, results, model_name=model_name, top_n=top_n)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

            if span and span.is_recording():
                span_set_input(span, query)
                span.set_attribute("reranking.output_count", len(results))
                span.set_attribute("reranking.latency_ms", elapsed_ms)
                if results:
                    scores = [
                        r.similarity_score for r in results if r.similarity_score is not None
                    ]
                    if scores:
                        span.set_attribute("reranking.top_score", round(max(scores), 4))
                        span.set_attribute("reranking.min_score", round(min(scores), 4))
                _set_result_attributes(span, results)
                span_set_output(
                    span,
                    {"count": len(results), "latency_ms": elapsed_ms},
                )

    return results


def _set_result_attributes(span, results: list[RetrievalResult]) -> None:
    """Set per-document attributes on a span for Phoenix retrieval UI."""
    if span is None or not span.is_recording():
        return
    for i, r in enumerate(results[:20]):  # Cap at 20 to avoid attribute bloat
        prefix = f"retrieval.documents.{i}"
        span.set_attribute(f"{prefix}.document.id", r.chunk_id or r.document_id or str(i))
        span.set_attribute(f"{prefix}.document.page", r.page_number)
        if r.similarity_score is not None:
            span.set_attribute(f"{prefix}.document.score", round(r.similarity_score, 4))
        if r.document_id:
            span.set_attribute(f"{prefix}.document.source", r.document_id)
