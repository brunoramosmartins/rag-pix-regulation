"""RAG pipeline orchestration."""

from dataclasses import dataclass
from typing import Callable

from src.retrieval.retriever import RetrievalResult, retrieve
from src.utils.document_aliases import get_document_alias

from .context_builder import build_context
from src.llm import LLMClient
from src.llm.llm_client import LLMUsage
from .prompt_template import build_prompt

try:
    from src.observability.tracing import span_set_input, span_set_output, trace_span
except ImportError:

    def trace_span(name: str, attributes=None, openinference_span_kind=None):
        from contextlib import nullcontext

        return nullcontext()

    def span_set_input(span, value): ...
    def span_set_output(span, value): ...


MAX_ATTR_LEN = 4000


def _truncate(s: str, max_len: int = MAX_ATTR_LEN) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + "...[truncated]"


def _build_citations(chunks: list[RetrievalResult]) -> list[str]:
    # Deduplicate by (resolved_alias, page) so that legacy and current document_ids
    # for the same document don't produce duplicate citation entries.
    seen: set[tuple[str, int]] = set()
    citations: list[str] = []
    for r in chunks:
        alias = get_document_alias(r.document_id)
        key = (alias, r.page_number)
        if key not in seen:
            seen.add(key)
            citations.append(f"{alias}, p. {r.page_number}")
    return citations


def _format_answer_with_citations(answer: str, citations: list[str]) -> str:
    """Append citation footer so the user knows where to verify the information."""
    if not citations:
        return answer
    refs = ", ".join(citations)
    footer = f"\n\n---\n*Fontes consultadas: {refs}. Para verificar, consulte os documentos originais citados.*"
    return answer.rstrip() + footer


@dataclass
class RAGResponse:
    """Structured RAG response with full trace for debugging and evaluation."""

    query: str
    answer: str
    context: str
    retrieved_chunks: list[RetrievalResult]
    citations: list[str]
    llm_usage: LLMUsage | None = None


def _load_config() -> dict:
    """Load full config from config.yaml. Returns empty dict on failure."""
    try:
        import yaml
        from pathlib import Path

        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def _default_retrieve(query: str, top_k: int) -> list[RetrievalResult]:
    cfg = _load_config().get("retrieval", {})
    min_similarity = cfg.get("min_similarity", 0.0)
    return retrieve(query, top_k=top_k, min_similarity=min_similarity)


def answer_query(
    query: str,
    llm: LLMClient,
    top_k: int = 5,
    max_chunks: int = 20,
    max_context_tokens: int | None = None,
    retriever: Callable[[str, int], list[RetrievalResult]] | None = None,
) -> RAGResponse:
    """
    Answer a query using RAG: retrieve → build context → prompt → generate.

    Returns RAGResponse with answer including citation footer for verification.

    When max_context_tokens is None, the value is loaded from config.yaml
    (rag.max_context_tokens). Falls back to 2048 if config is unavailable.
    """
    retrieve_fn = retriever or _default_retrieve

    if max_context_tokens is None:
        cfg = _load_config().get("rag", {})
        max_context_tokens = cfg.get("max_context_tokens", 2048)

    with trace_span("rag_pipeline", openinference_span_kind="chain") as parent_span:
        span_set_input(parent_span, query)
        if parent_span and parent_span.is_recording():
            parent_span.set_attribute("rag.top_k", top_k)
            parent_span.set_attribute("rag.max_chunks", max_chunks)
            parent_span.set_attribute("rag.max_context_tokens", max_context_tokens)

        with trace_span("retrieval", openinference_span_kind="retriever") as span:
            chunks = retrieve_fn(query, top_k)
            if span and span.is_recording():
                span_set_input(span, query)
                span.set_attribute("retrieval.top_k", top_k)
                span.set_attribute("retrieval.result_count", len(chunks))
                # Unique documents and pages for quick overview
                unique_docs = {c.document_id for c in chunks if c.document_id}
                unique_pages = {(c.document_id, c.page_number) for c in chunks}
                span.set_attribute("retrieval.unique_documents", len(unique_docs))
                span.set_attribute("retrieval.unique_pages", len(unique_pages))
                if chunks:
                    scores = [c.similarity_score for c in chunks if c.similarity_score is not None]
                    if scores:
                        span.set_attribute("retrieval.score.max", round(max(scores), 4))
                        span.set_attribute("retrieval.score.min", round(min(scores), 4))
                        span.set_attribute("retrieval.score.mean", round(sum(scores) / len(scores), 4))
                for i, c in enumerate(chunks):
                    alias = get_document_alias(c.document_id)
                    span.set_attribute(f"retrieval.documents.{i}.document.id", c.chunk_id or c.document_id or str(i))
                    span.set_attribute(f"retrieval.documents.{i}.document.content", _truncate(c.text))
                    if c.similarity_score is not None:
                        span.set_attribute(f"retrieval.documents.{i}.document.score", round(c.similarity_score, 4))
                    meta = f"source={alias}, page={c.page_number}"
                    if c.section_title:
                        meta += f", section={c.section_title}"
                    span.set_attribute(f"retrieval.documents.{i}.document.metadata", meta)
                span_set_output(
                    span,
                    {
                        "count": len(chunks),
                        "refs": [
                            f"{get_document_alias(c.document_id)}, p. {c.page_number}"
                            for c in chunks
                        ],
                    },
                )

        with trace_span("context_building", openinference_span_kind="chain") as span:
            context = build_context(
                chunks, max_chunks=max_chunks, max_tokens=max_context_tokens
            )
            if span and span.is_recording():
                span_set_input(span, {"chunk_count": len(chunks), "max_tokens": max_context_tokens})
                span.set_attribute("context.char_count", len(context))
                span.set_attribute("context.chunk_count", len(chunks))
                span_set_output(span, _truncate(context))

        with trace_span(
            "prompt_construction", openinference_span_kind="prompt"
        ) as span:
            prompt = build_prompt(context, query)
            if span and span.is_recording():
                span_set_input(
                    span, {"context_len": len(context), "query": query[:200]}
                )
                span.set_attribute("prompt.char_count", len(prompt))
                span_set_output(span, _truncate(prompt))

        with trace_span("llm_generation", openinference_span_kind="llm") as span:
            if span and span.is_recording():
                model_name = getattr(llm, "model", "unknown")
                span.set_attribute("llm.model_name", model_name)
                span.set_attribute("llm.invocation_parameters", _truncate(prompt))
                # Capture LLM config for debugging
                for attr_name in ("temperature", "top_p", "num_ctx", "num_predict"):
                    val = getattr(llm, attr_name, None)
                    if val is not None:
                        span.set_attribute(f"llm.{attr_name}", val)
            answer, usage = llm.generate(prompt)
            citations = _build_citations(chunks)
            if span and span.is_recording():
                span.set_attribute("llm.response_length", len(answer))
                span.set_attribute("llm.citation_count", len(citations))
                span_set_output(span, {"answer": _truncate(answer), "citations": citations})
                if usage:
                    span.set_attribute("llm.token_count.prompt", usage.prompt_tokens)
                    span.set_attribute("llm.token_count.completion", usage.completion_tokens)
                    span.set_attribute("llm.token_count.total", usage.total_tokens)

        answer_with_citations = _format_answer_with_citations(answer, citations)
        if parent_span and parent_span.is_recording():
            parent_span.set_attribute("rag.citation_count", len(citations))
            parent_span.set_attribute("rag.answer_length", len(answer_with_citations))
            span_set_output(
                parent_span, {"answer": answer_with_citations, "citations": citations}
            )

    return RAGResponse(
        query=query,
        answer=answer_with_citations,
        context=context,
        retrieved_chunks=chunks,
        citations=citations,
        llm_usage=usage,
    )
