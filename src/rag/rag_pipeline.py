"""RAG pipeline orchestration."""

from dataclasses import dataclass
from typing import Callable

from src.retrieval.retriever import RetrievalResult, retrieve

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
    """Build deterministic citations from retrieved chunks."""
    seen: set[tuple[str, int]] = set()
    citations: list[str] = []
    for r in chunks:
        key = (r.document_id, r.page_number)
        if key not in seen:
            seen.add(key)
            citations.append(f"{r.document_id} p.{r.page_number}")
    return citations


def _format_answer_with_citations(answer: str, citations: list[str]) -> str:
    """Append citation footer so the user knows where to verify the information."""
    if not citations:
        return answer

    refs = ", ".join(citations)

    footer = (
        "\n\n---\n"
        f"*Fontes consultadas: {refs}. "
        "Para verificar, consulte os documentos originais citados.*"
    )

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


def _default_retrieve(query: str, top_k: int) -> list[RetrievalResult]:
    return retrieve(query, top_k=top_k)


def answer_query(
    query: str,
    llm: LLMClient,
    top_k: int = 5,
    max_chunks: int = 5,
    max_context_tokens: int | None = 4096,
    retriever: Callable[[str, int], list[RetrievalResult]] | None = None,
) -> RAGResponse:
    """
    Answer a query using RAG: retrieve → build context → prompt → generate.
    """

    retrieve_fn = retriever or _default_retrieve

    with trace_span("rag_pipeline", openinference_span_kind="chain") as parent_span:
        span_set_input(parent_span, query)

        with trace_span("retrieval", openinference_span_kind="retriever") as span:
            chunks = retrieve_fn(query, top_k)

            if span and span.is_recording():
                span_set_input(span, query)

                for i, c in enumerate(chunks):
                    doc_id = c.chunk_id or c.document_id or str(i)

                    span.set_attribute(f"retrieval.documents.{i}.document.id", doc_id)
                    span.set_attribute(
                        f"retrieval.documents.{i}.document.content",
                        _truncate(c.text),
                    )

                    meta = f"document_id={c.document_id}, page={c.page_number}"

                    if c.section_title:
                        meta += f", section={c.section_title}"

                    span.set_attribute(
                        f"retrieval.documents.{i}.document.metadata", meta
                    )

                span_set_output(
                    span,
                    {
                        "count": len(chunks),
                        "refs": [f"{c.document_id} p.{c.page_number}" for c in chunks],
                    },
                )

        with trace_span("context_building", openinference_span_kind="chain") as span:
            context = build_context(chunks, max_chunks=max_chunks, max_tokens=max_context_tokens)

            if span and span.is_recording():
                span_set_input(span, {"chunk_count": len(chunks)})
                span_set_output(span, _truncate(context))

        with trace_span("prompt_construction", openinference_span_kind="prompt") as span:
            prompt = build_prompt(context, query)

            if span and span.is_recording():
                span_set_input(span, {"context_len": len(context), "query": query[:200]})
                span_set_output(span, _truncate(prompt))

        with trace_span("llm_generation", openinference_span_kind="llm") as span:
            answer, usage = llm.generate(prompt)

            citations = _build_citations(chunks)

            if span and span.is_recording():
                span_set_output(
                    span,
                    {
                        "answer": _truncate(answer),
                        "citations": citations,
                    },
                )

                if usage:
                    span.set_attribute("llm.token_count.prompt", usage.prompt_tokens)
                    span.set_attribute("llm.token_count.completion", usage.completion_tokens)

        answer_with_citations = _format_answer_with_citations(answer, citations)

        if parent_span and parent_span.is_recording():
            span_set_output(
                parent_span,
                {"answer": answer_with_citations, "citations": citations},
            )

    return RAGResponse(
        query=query,
        answer=answer_with_citations,
        context=context,
        retrieved_chunks=chunks,
        citations=citations,
        llm_usage=usage,
    )