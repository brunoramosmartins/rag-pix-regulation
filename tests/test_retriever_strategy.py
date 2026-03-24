"""Unit tests for retriever search strategy dispatch.

Uses patch.object on directly-imported modules to avoid triggering
the sentence_transformers -> tf_keras import chain on Windows.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.retrieval.models import RetrievalResult
from tests.helpers import make_test_settings

# These modules DON'T trigger sentence_transformers import chain
import src.retrieval.retriever as retriever_module
import src.retrieval.keyword_search as ks_module
import src.retrieval.hybrid_search as hs_module
import src.retrieval.vector_search as vs_module


def _make_raw_result(chunk_id: str = "c1", score: float = 0.8) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": "doc",
        "page_number": 1,
        "section_title": None,
        "text": f"text for {chunk_id}",
        "source_file": "test.pdf",
        "similarity_score": score,
    }


VECTOR_SETTINGS = make_test_settings(search_strategy="vector")
KEYWORD_SETTINGS = make_test_settings(search_strategy="keyword")
HYBRID_SETTINGS = make_test_settings(
    search_strategy="hybrid", hybrid_alpha=0.6, hybrid_fusion_type="ranked",
)


def test_retrieve_vector_strategy_calls_vector_search() -> None:
    """Vector strategy calls embed_query + vector_search."""
    mock_embed = MagicMock(return_value=[0.0] * 1024)
    with (
        patch.object(retriever_module, "_get_settings", return_value=VECTOR_SETTINGS),
        patch.object(vs_module, "vector_search", return_value=[_make_raw_result()]) as mock_vs,
        patch.dict("sys.modules", {"src.retrieval.query_embedding": MagicMock(embed_query=mock_embed)}),
    ):
        results = retriever_module.retrieve("test query", top_k=5, search_strategy="vector")
        mock_embed.assert_called_once_with("test query")
        mock_vs.assert_called_once()
        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)


def test_retrieve_keyword_strategy_calls_keyword_search() -> None:
    """Keyword strategy calls keyword_search, no embedding generated."""
    with (
        patch.object(retriever_module, "_get_settings", return_value=KEYWORD_SETTINGS),
        patch.object(ks_module, "keyword_search", return_value=[_make_raw_result()]) as mock_ks,
    ):
        results = retriever_module.retrieve("Art. 3", top_k=5, search_strategy="keyword")
        mock_ks.assert_called_once()
        assert len(results) == 1


def test_retrieve_hybrid_strategy_calls_hybrid_search() -> None:
    """Hybrid strategy calls embed_query + hybrid_search."""
    mock_embed = MagicMock(return_value=[0.0] * 1024)
    with (
        patch.object(retriever_module, "_get_settings", return_value=HYBRID_SETTINGS),
        patch.object(hs_module, "hybrid_search", return_value=[_make_raw_result()]) as mock_hs,
        patch.dict("sys.modules", {"src.retrieval.query_embedding": MagicMock(embed_query=mock_embed)}),
    ):
        results = retriever_module.retrieve("chave Pix", top_k=5, search_strategy="hybrid")
        mock_embed.assert_called_once_with("chave Pix")
        mock_hs.assert_called_once()
        call_kwargs = mock_hs.call_args.kwargs
        assert call_kwargs["alpha"] == 0.6
        assert call_kwargs["fusion_type"] == "ranked"
        assert len(results) == 1


def test_retrieve_default_strategy_is_vector() -> None:
    """When no strategy override is passed, uses config value (defaults to vector in test)."""
    mock_embed = MagicMock(return_value=[0.0] * 1024)
    with (
        patch.object(retriever_module, "_get_settings", return_value=VECTOR_SETTINGS),
        patch.object(vs_module, "vector_search", return_value=[_make_raw_result()]) as mock_vs,
        patch.dict("sys.modules", {"src.retrieval.query_embedding": MagicMock(embed_query=mock_embed)}),
    ):
        results = retriever_module.retrieve("test", top_k=5)
        mock_vs.assert_called_once()
        assert len(results) == 1


def test_retrieve_parameter_overrides_config() -> None:
    """Explicit search_strategy param overrides config.yaml value."""
    mock_embed = MagicMock(return_value=[0.0] * 1024)
    with (
        patch.object(retriever_module, "_get_settings", return_value=HYBRID_SETTINGS),
        patch.object(vs_module, "vector_search", return_value=[_make_raw_result()]) as mock_vs,
        patch.dict("sys.modules", {"src.retrieval.query_embedding": MagicMock(embed_query=mock_embed)}),
    ):
        # Config says "hybrid" but param says "vector"
        retriever_module.retrieve("test", top_k=5, search_strategy="vector")
        mock_vs.assert_called_once()


def test_retrieve_min_similarity_skipped_for_keyword() -> None:
    """min_similarity is not applied for keyword strategy (BM25 scores differ)."""
    with (
        patch.object(retriever_module, "_get_settings", return_value=KEYWORD_SETTINGS),
        patch.object(ks_module, "keyword_search", return_value=[_make_raw_result()]),
    ):
        # min_similarity=0.9 would filter out score=0.8 for vector, but keyword ignores it
        results = retriever_module.retrieve("test", top_k=5, search_strategy="keyword", min_similarity=0.9)
        assert len(results) == 1


def test_retrieve_alpha_parameter_overrides_config() -> None:
    """Alpha parameter overrides config hybrid.alpha value."""
    mock_embed = MagicMock(return_value=[0.0] * 1024)
    with (
        patch.object(retriever_module, "_get_settings", return_value=HYBRID_SETTINGS),
        patch.object(hs_module, "hybrid_search", return_value=[_make_raw_result()]) as mock_hs,
        patch.dict("sys.modules", {"src.retrieval.query_embedding": MagicMock(embed_query=mock_embed)}),
    ):
        retriever_module.retrieve("test", top_k=5, search_strategy="hybrid", alpha=0.3)
        call_kwargs = mock_hs.call_args.kwargs
        assert call_kwargs["alpha"] == 0.3  # param, not config's 0.6


def test_retrieve_invalid_strategy_raises() -> None:
    """Invalid search strategy raises ValueError."""
    with patch.object(retriever_module, "_get_settings", return_value=VECTOR_SETTINGS):
        with pytest.raises(ValueError, match="Invalid search_strategy"):
            retriever_module.retrieve("test", top_k=5, search_strategy="invalid")
