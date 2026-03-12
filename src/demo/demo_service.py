"""Demo service layer — RAG and baseline query execution for the Streamlit interface."""

import time
from typing import Any

from src.utils.document_aliases import get_document_alias
from src.utils.system_checks import check_rag_dependencies


def _to_baseline_dict(answer: str, latency_ms: int, model: str) -> dict[str, Any]:
    """Build standardized baseline response dict for UI."""
    return {
        "answer": answer,
        "latency_ms": latency_ms,
        "sources": 0,
        "model": model,
        "chunks": [],
        "citations": [],
    }


def _to_rag_dict(
    answer: str,
    latency_ms: int,
    model: str,
    chunks: list[dict[str, Any]],
    citations: list[str],
) -> dict[str, Any]:
    """Build standardized RAG response dict for UI."""
    return {
        "answer": answer,
        "latency_ms": latency_ms,
        "sources": len(chunks),
        "model": model,
        "chunks": chunks,
        "citations": citations,
    }


def run_baseline_query(query: str) -> dict[str, Any]:
    """
    Generate a response without retrieval (baseline LLM only).

    Demonstrates hallucination risk when no regulatory context is provided.
    """
    from src.llm import BaselineLLM

    prompt = f"""Responda à seguinte pergunta sobre regulamentação Pix do Banco Central do Brasil.

Pergunta: {query}

Responda com base no seu conhecimento. Se não souber, indique que a informação não está disponível."""

    llm = BaselineLLM()
    start = time.perf_counter()
    answer, _ = llm.generate(prompt)
    latency_ms = int((time.perf_counter() - start) * 1000)

    return _to_baseline_dict(answer, latency_ms, "llama3.2:3b")


def run_rag_query(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    Generate a response using the RAG pipeline (retrieve → context → prompt → LLM).

    Returns grounded answer with citations and retrieved chunks.
    """
    from src.llm import BaselineLLM
    from src.rag import answer_query

    llm = BaselineLLM()
    start = time.perf_counter()
    response = answer_query(query, llm=llm, top_k=top_k)
    latency_ms = int((time.perf_counter() - start) * 1000)

    chunks = [
        {
            "document_id": c.document_id,
            "document_alias": get_document_alias(c.document_id),
            "page": c.page_number,
            "section": c.section_title or "—",
            "text": c.text[:500] + ("..." if len(c.text) > 500 else ""),
            "score": c.similarity_score or 0.0,
        }
        for c in response.retrieved_chunks
    ]

    return _to_rag_dict(
        response.answer,
        latency_ms,
        "llama3.2:3b + RAG",
        chunks,
        response.citations,
    )


def get_demo_health() -> tuple[bool, str]:
    """
    Check if demo dependencies (Weaviate, Ollama) are ready.

    Returns (ready, message).
    """
    return check_rag_dependencies()
