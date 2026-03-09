"""RAG pipeline orchestration."""

from dataclasses import dataclass
from typing import Callable

from src.retrieval.models import RetrievalResult

from .context_builder import build_context
from src.llm import LLMClient
from .prompt_template import build_prompt


def _default_retrieve(query: str, top_k: int) -> list[RetrievalResult]:
    from src.retrieval.retriever import retrieve
    return retrieve(query, top_k=top_k)


@dataclass
class RAGResponse:
    """Structured RAG response with full trace for debugging and evaluation."""

    query: str
    answer: str
    context: str
    retrieved_chunks: list[RetrievalResult]
    citations: list[str]


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

    Parameters
    ----------
    query : str
        User question.
    llm : LLMClient
        LLM client (RAG pipeline does not depend on concrete implementation).
    top_k : int
        Number of chunks to retrieve.
    max_chunks : int
        Maximum chunks to include in context.
    max_context_tokens : int | None
        Token limit for context. Truncates if exceeded.

    Returns
    -------
    RAGResponse
        Answer, context, chunks, and deterministic citations.
    """
    retrieve_fn = retriever or _default_retrieve
    chunks = retrieve_fn(query, top_k)
    context = build_context(
        chunks,
        max_chunks=max_chunks,
        max_tokens=max_context_tokens,
    )
    prompt = build_prompt(context, query)
    answer = llm.generate(prompt)

    citations = _build_citations(chunks)

    return RAGResponse(
        query=query,
        answer=answer,
        context=context,
        retrieved_chunks=chunks,
        citations=citations,
    )
