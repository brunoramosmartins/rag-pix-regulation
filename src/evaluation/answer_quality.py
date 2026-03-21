"""Answer quality metrics: semantic similarity and concept coverage."""

import logging
import unicodedata
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnswerQualityResult:
    """Answer quality evaluation result."""

    answer_similarity: float
    concept_coverage: float
    quality_score: float


def _normalize_text(text: str) -> str:
    """Lowercase and remove accents for fuzzy matching."""
    text = text.lower()
    # NFD decomposes accented chars, then strip combining marks
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def compute_answer_similarity(
    generated: str,
    expected: str,
    model: object | None = None,
) -> float:
    """
    Compute cosine similarity between generated answer and expected summary.

    Uses the cached BGE-M3 embedding model for encoding.
    Returns a float in [0, 1]. Returns 0.0 if either string is empty.

    Parameters
    ----------
    model : object | None
        Optional pre-loaded SentenceTransformer model. If None, loads BGE-M3.
    """
    if not generated.strip() or not expected.strip():
        return 0.0

    try:
        if model is None:
            from src.embeddings.embedding_generator import get_embedding_model

            model = get_embedding_model()

        embeddings = model.encode(
            [generated, expected],
            normalize_embeddings=True,
        )
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        return max(0.0, round(similarity, 4))
    except Exception as e:
        logger.warning("Failed to compute answer similarity: %s", e)
        return 0.0


def compute_concept_coverage(answer: str, key_concepts: list[str]) -> float:
    """
    Fraction of key concepts found in the answer.

    Uses case-insensitive, accent-insensitive substring matching.
    Returns 1.0 if key_concepts is empty (nothing required).
    """
    if not key_concepts:
        return 1.0

    normalized_answer = _normalize_text(answer)
    matched = sum(
        1 for concept in key_concepts
        if _normalize_text(concept) in normalized_answer
    )
    return round(matched / len(key_concepts), 4)


def compute_answer_quality(
    generated: str,
    expected: str,
    key_concepts: list[str],
) -> AnswerQualityResult:
    """
    Compute combined answer quality score.

    Combines embedding similarity (60% weight) and concept coverage (40% weight).
    """
    similarity = compute_answer_similarity(generated, expected)
    coverage = compute_concept_coverage(generated, key_concepts)
    quality = round(0.6 * similarity + 0.4 * coverage, 4)

    return AnswerQualityResult(
        answer_similarity=similarity,
        concept_coverage=coverage,
        quality_score=quality,
    )
