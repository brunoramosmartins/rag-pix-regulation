"""Unit tests for answer quality metrics."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.evaluation.answer_quality import (
    AnswerQualityResult,
    _normalize_text,
    compute_answer_quality,
    compute_answer_similarity,
    compute_concept_coverage,
)


# --- Concept coverage tests (no mocking needed) ---


def test_concept_coverage_all_present() -> None:
    """All concepts found returns 1.0."""
    answer = "O bloqueio de recursos é feito via rastreamento de transações."
    concepts = ["bloqueio de recursos", "rastreamento de transações"]
    assert compute_concept_coverage(answer, concepts) == 1.0


def test_concept_coverage_none_present() -> None:
    """No concepts found returns 0.0."""
    answer = "O sistema Pix permite pagamentos instantâneos."
    concepts = ["bloqueio de recursos", "rastreamento de transações"]
    assert compute_concept_coverage(answer, concepts) == 0.0


def test_concept_coverage_partial() -> None:
    """1 of 4 concepts returns 0.25."""
    answer = "O bloqueio de recursos é feito automaticamente."
    concepts = ["bloqueio de recursos", "rastreamento", "devolução", "fraude"]
    assert compute_concept_coverage(answer, concepts) == 0.25


def test_concept_coverage_accent_insensitive() -> None:
    """Accent-insensitive matching: 'transacao' matches 'transação'."""
    answer = "A transacao foi realizada com sucesso via recuperacao."
    concepts = ["transação", "recuperação"]
    assert compute_concept_coverage(answer, concepts) == 1.0


def test_concept_coverage_empty_concepts() -> None:
    """Empty concept list returns 1.0 (nothing required)."""
    assert compute_concept_coverage("any answer", []) == 1.0


def test_normalize_text_removes_accents() -> None:
    """_normalize_text removes Portuguese accents."""
    assert _normalize_text("transação") == "transacao"
    assert _normalize_text("Recuperação") == "recuperacao"
    assert _normalize_text("Não está disponível") == "nao esta disponivel"


# --- Answer similarity tests (inject mock model via parameter) ---


def _make_mock_model(embeddings: list) -> MagicMock:
    """Create a mock SentenceTransformer that returns pre-defined embeddings."""
    model = MagicMock()
    model.encode.return_value = np.array(embeddings)
    return model


def test_answer_similarity_identical() -> None:
    """Identical strings should return similarity close to 1.0."""
    vec = np.array([1.0, 0.0, 0.0])
    norm = vec / np.linalg.norm(vec)
    mock_model = _make_mock_model([norm, norm])

    result = compute_answer_similarity("test", "test", model=mock_model)
    assert result >= 0.99


def test_answer_similarity_empty() -> None:
    """Empty generated answer returns 0.0 without calling model."""
    result = compute_answer_similarity("", "expected answer")
    assert result == 0.0


def test_answer_similarity_orthogonal() -> None:
    """Orthogonal vectors should return 0.0."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    mock_model = _make_mock_model([v1, v2])

    result = compute_answer_similarity("text a", "text b", model=mock_model)
    assert result == 0.0


# --- Quality score tests ---


def test_quality_score_weighted() -> None:
    """Quality score uses 0.6 * similarity + 0.4 * coverage."""
    vec = np.array([1.0, 0.0])
    norm = vec / np.linalg.norm(vec)
    mock_model = _make_mock_model([norm, norm])

    with patch(
        "src.evaluation.answer_quality.compute_answer_similarity",
        side_effect=lambda g, e, model=None: compute_answer_similarity(
            g, e, model=mock_model
        ),
    ):
        result = compute_answer_quality(
            generated="bloqueio de recursos",
            expected="bloqueio de recursos",
            key_concepts=["bloqueio de recursos", "rastreamento"],
        )
        assert isinstance(result, AnswerQualityResult)
        # similarity ~1.0, coverage = 1/2 = 0.5
        # quality = 0.6 * 1.0 + 0.4 * 0.5 = 0.8
        assert abs(result.quality_score - 0.8) < 0.01
        assert result.concept_coverage == 0.5
        assert result.answer_similarity >= 0.99
