"""RAG pipeline orchestration."""

import json
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

# OpenInference attribute names for Phoenix
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"
SPAN_KIND = "openinference.span.kind"

MAX_ATTR_LEN = 4000


def _truncate(s: str, max_len: int = MAX_ATTR_LEN) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + "...[truncated]"


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

    with trace_span("retrieval", {INPUT_VALUE: query, SPAN_KIND: "RETRIEVER"}) as span:
        chunks = retrieve_fn(query, top_k)
        if span and span.is_recording():
            refs = [f"{c.document_id} p.{c.page_number}" for c in chunks]
            span.set_attribute(OUTPUT_VALUE, json.dumps({"count": len(chunks), "refs": refs}))

    with trace_span("context_building", {SPAN_KIND: "CHAIN"}) as span:
        context = build_context(chunks, max_chunks=max_chunks, max_tokens=max_context_tokens)
        if span and span.is_recording():
            span.set_attribute(INPUT_VALUE, json.dumps({"chunk_count": len(chunks)}))
            span.set_attribute(OUTPUT_VALUE, _truncate(context))

    with trace_span("prompt_construction", {SPAN_KIND: "PROMPT"}) as span:
        prompt = build_prompt(context, query)
        if span and span.is_recording():
            span.set_attribute(INPUT_VALUE, _truncate(f"context_len={len(context)}, query={query[:200]}"))
            span.set_attribute(OUTPUT_VALUE, _truncate(prompt))

    with trace_span("llm_generation", {INPUT_VALUE: _truncate(prompt), SPAN_KIND: "LLM"}) as span:
        answer, usage = llm.generate(prompt)
        if span and span.is_recording():
            span.set_attribute(OUTPUT_VALUE, _truncate(answer))
            if usage:
                span.set_attribute("llm.token_count.prompt", usage.prompt_tokens)
                span.set_attribute("llm.token_count.completion", usage.completion_tokens)
    citations = _build_citations(chunks)

    return RAGResponse(
        query=query,
        answer=answer,
        context=context,
        retrieved_chunks=chunks,
        citations=citations,
        llm_usage=usage,
    )
