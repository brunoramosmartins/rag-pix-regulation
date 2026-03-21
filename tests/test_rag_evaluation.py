"""Unit tests for RAG evaluation metrics."""

from dataclasses import dataclass

from src.evaluation.rag_evaluation import (
    compute_citation_coverage,
    detect_hallucination,
    evaluate_rag_response,
)


@dataclass
class MockChunk:
    document_id: str
    page_number: int


def test_compute_citation_coverage_full() -> None:
    """Citation coverage = 1 when all citations match retrieved chunks."""
    citations = ["manual_dict p.122", "manual_dict p.123"]
    chunks = [MockChunk("manual_dict", 122), MockChunk("manual_dict", 123)]
    assert compute_citation_coverage(citations, chunks) == 1.0


def test_compute_citation_coverage_partial() -> None:
    """Citation coverage = 0.5 when half match."""
    citations = ["manual_dict p.122", "manual_dict p.999"]
    chunks = [MockChunk("manual_dict", 122)]
    assert compute_citation_coverage(citations, chunks) == 0.5


def test_compute_citation_coverage_empty_citations() -> None:
    """Empty citations returns 1.0 (nothing to validate)."""
    assert compute_citation_coverage([], [MockChunk("x", 1)]) == 1.0


def test_detect_hallucination_abstention() -> None:
    """Answer saying info not available is not hallucination."""
    assert (
        detect_hallucination(
            "A informacao nao esta disponivel no contexto.", "short ctx", ""
        )
        is False
    )


def test_detect_hallucination_long_answer_short_context() -> None:
    """Long answer with minimal context is suspicious."""
    long_answer = "x" * 300
    short_context = "y" * 50
    assert detect_hallucination(long_answer, short_context, "") is True


def test_evaluate_rag_response() -> None:
    """evaluate_rag_response produces RAGEvaluationResult."""
    result = evaluate_rag_response(
        query_id="q1",
        answer="Resposta baseada no contexto.",
        context="Contexto regulatório aqui.",
        citations=["manual_dict p.1"],
        retrieved_chunks=[MockChunk("manual_dict", 1)],
        expected_pages={1},
        precision_at_k=1.0,
        recall_at_k=1.0,
    )
    assert result.query_id == "q1"
    assert result.citation_coverage == 1.0
    assert result.hallucination_detected is False


def test_evaluate_rag_response_backward_compat() -> None:
    """Without answer quality params, new fields default to 0.0."""
    result = evaluate_rag_response(
        query_id="q1",
        answer="Resposta.",
        context="Contexto.",
        citations=[],
        retrieved_chunks=[],
        expected_pages={1},
        precision_at_k=0.5,
        recall_at_k=0.5,
    )
    assert result.answer_similarity == 0.0
    assert result.concept_coverage == 0.0
    assert result.quality_score == 0.0


def test_evaluate_rag_response_with_answer_quality() -> None:
    """With expected_answer_summary, answer quality fields are populated."""
    from unittest.mock import patch, MagicMock
    import numpy as np
    from src.evaluation.answer_quality import compute_answer_similarity as _real_sim

    vec = np.array([1.0, 0.0])
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([vec, vec])

    with patch(
        "src.evaluation.answer_quality.compute_answer_similarity",
        side_effect=lambda g, e, model=None: _real_sim(g, e, model=mock_model),
    ):
        result = evaluate_rag_response(
            query_id="q1",
            answer="O bloqueio de recursos é feito automaticamente.",
            context="Contexto regulatório aqui.",
            citations=[],
            retrieved_chunks=[],
            expected_pages={1},
            precision_at_k=0.5,
            recall_at_k=0.5,
            expected_answer_summary="O bloqueio de recursos protege contra fraudes.",
            key_concepts=["bloqueio de recursos", "fraudes"],
        )
        assert result.answer_similarity > 0.0
        assert result.concept_coverage > 0.0
        assert result.quality_score > 0.0
