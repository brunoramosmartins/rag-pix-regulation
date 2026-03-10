"""RAG evaluation: groundedness, citation coverage, hallucination detection."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class RAGEvaluationResult:
    """Structured RAG evaluation result per query."""

    query_id: str
    precision_at_k: float
    recall_at_k: float
    citation_coverage: float
    groundedness_score: float
    hallucination_detected: bool
    context_used: bool


def compute_citation_coverage(
    citations: list[str],
    retrieved_chunks: list[Any],
) -> float:
    """
    Citation coverage: fraction of citations that appear in retrieved chunks.

    Citations are deterministic (e.g. "manual_dict p.122").
    Coverage = 1.0 if all citations match retrieved (doc_id, page).
    """
    if not citations:
        return 1.0
    valid_refs = {
        (r.document_id, r.page_number) if hasattr(r, "document_id") else (r.get("document_id", ""), r.get("page_number", 0))
        for r in retrieved_chunks
    }
    covered = 0
    for c in citations:
        # Parse "doc_id p.123" or similar
        parts = c.split()
        if len(parts) >= 2 and parts[-1].startswith("p."):
            try:
                page = int(parts[-1].replace("p.", ""))
                doc_id = " ".join(parts[:-1]).strip()
                if (doc_id, page) in valid_refs:
                    covered += 1
            except ValueError:
                pass
        else:
            covered += 1  # Assume valid if we can't parse
    return covered / len(citations) if citations else 1.0


def detect_hallucination(
    answer: str,
    context: str,
    query: str,
) -> bool:
    """
    Heuristic hallucination detection: answer claims info not in context.

    Returns True if hallucination detected.
    """
    answer_lower = answer.lower()
    context_lower = context.lower()

    # Good: explicitly says info not available
    no_info_phrases = [
        "não disponível",
        "não está disponível",
        "nao disponivel",
        "nao esta disponivel",
        "não é possível determinar",
        "não contém",
        "não está presente",
        "informação não disponível",
        "não encontrei",
        "não consta",
    ]
    if any(p in answer_lower for p in no_info_phrases):
        return False  # Correctly abstained

    # Bad: long answer with empty/minimal context
    if len(context.strip()) < 100 and len(answer.strip()) > 200:
        return True

    # Simple check: answer length vs context (very long answer with short context = suspicious)
    if context and len(answer) > 3 * len(context):
        return True

    return False


def check_context_usage(answer: str, context: str) -> bool:
    """Whether response appears to reference the provided context."""
    if not context or not answer:
        return False
    # Heuristic: answer should not be generic when context exists
    generic_phrases = [
        "não tenho informações",
        "não posso ajudar",
        "como modelo de linguagem",
    ]
    return not any(p in answer.lower() for p in generic_phrases)


def evaluate_rag_response(
    query_id: str,
    answer: str,
    context: str,
    citations: list[str],
    retrieved_chunks: list[Any],
    expected_pages: set[int],
    precision_at_k: float,
    recall_at_k: float,
) -> RAGEvaluationResult:
    """
    Compute full RAG evaluation for a single response.
    """
    citation_cov = compute_citation_coverage(citations, retrieved_chunks)
    hallucination = detect_hallucination(answer, context, "")
    context_used = check_context_usage(answer, context)
    groundedness = citation_cov * (1.0 if context_used else 0.5) * (0.0 if hallucination else 1.0)

    return RAGEvaluationResult(
        query_id=query_id,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        citation_coverage=citation_cov,
        groundedness_score=groundedness,
        hallucination_detected=hallucination,
        context_used=context_used,
    )
