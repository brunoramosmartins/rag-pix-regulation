"""RAG pipeline orchestration."""

from dataclasses import dataclass
from typing import Callable

from src.retrieval.retriever import RetrievalResult, retrieve

from .context_builder import build_context
from src.llm import LLMClient
from src.llm.llm_client import LLMUsage
from .prompt_template import build_prompt

try:
    from src.observability.tracing import trace_span
except ImportError:
    def trace_span(name: str, attributes=None):
        from contextlib import nullcontext
        return nullcontext()


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


def _build_citations(chunks: list[RetrievalResult]) -> list[str]:
    seen: set[tuple[str, int]] = set()
    citations: list[str] = []
    for r in chunks:
        key = (r.document_id, r.page_number)
        if key not in seen:
            seen.add(key)
            citations.append(f"{r.document_id} p.{r.page_number}")
    return citations


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
    with trace_span("retrieval", {"query": query[:100]}):
        chunks = retrieve_fn(query, top_k)
    with trace_span("context_building"):
        context = build_context(chunks, max_chunks=max_chunks, max_tokens=max_context_tokens)
    with trace_span("prompt_construction"):
        prompt = build_prompt(context, query)
    with trace_span("llm_generation", {"query": query[:100]}):
        answer, usage = llm.generate(prompt)
    citations = _build_citations(chunks)

    return RAGResponse(
        query=query,
        answer=answer,
        context=context,
        retrieved_chunks=chunks,
        citations=citations,
        llm_usage=usage,
    )
