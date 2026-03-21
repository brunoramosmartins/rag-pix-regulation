"""Tests for granular tracing spans in retrieval pipeline and enriched RAG spans.

Verifies that:
- retriever.py creates inner spans for embedding, search, and reranking
- rag_pipeline.py sets enriched attributes on existing spans
- All spans are safe to use when tracing is unavailable (no-op fallback)
"""

from unittest.mock import MagicMock, patch

import src.retrieval.retriever as retriever_module
import src.retrieval.keyword_search as ks_module
import src.retrieval.hybrid_search as hs_module
import src.retrieval.vector_search as vs_module
from src.retrieval.models import RetrievalResult


def _make_raw_result(chunk_id: str = "c1", score: float = 0.8) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": "doc",
        "page_number": 1,
        "section_title": "Sec",
        "text": f"text for {chunk_id}",
        "source_file": "test.pdf",
        "similarity_score": score,
    }


VECTOR_CONFIG = {
    "retrieval": {"search_strategy": "vector", "min_similarity": 0.0},
    "reranking": {"enabled": False},
}

KEYWORD_CONFIG = {
    "retrieval": {"search_strategy": "keyword"},
    "reranking": {"enabled": False},
}

HYBRID_CONFIG = {
    "retrieval": {
        "search_strategy": "hybrid",
        "hybrid": {"alpha": 0.5, "fusion_type": "ranked"},
    },
    "reranking": {"enabled": False},
}

RERANK_CONFIG = {
    "retrieval": {"search_strategy": "keyword"},
    "reranking": {
        "enabled": True,
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_n": 2,
    },
}


# --- Issue 1: Granular tracing spans in retrieval pipeline ---


class TestRetrieverTracingSpans:
    """Verify that retrieve() creates inner spans for each substep."""

    def test_vector_strategy_creates_embedding_and_search_spans(self) -> None:
        """Vector strategy creates query_embedding + vector_search spans."""
        mock_embed = MagicMock(return_value=[0.0] * 1024)
        span_names: list[str] = []

        from contextlib import contextmanager

        @contextmanager
        def _capture_span(name, attributes=None, openinference_span_kind=None):
            span_names.append(name)
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            yield mock_span

        with (
            patch.object(retriever_module, "_load_config", return_value=VECTOR_CONFIG),
            patch.object(vs_module, "vector_search", return_value=[_make_raw_result()]),
            patch.dict("sys.modules", {"src.retrieval.query_embedding": MagicMock(embed_query=mock_embed)}),
            patch.object(retriever_module, "trace_span", side_effect=_capture_span),
        ):
            retriever_module.retrieve("test", top_k=5, search_strategy="vector")

        assert "query_embedding" in span_names
        assert "vector_search" in span_names

    def test_keyword_strategy_creates_keyword_search_span(self) -> None:
        """Keyword strategy creates keyword_search span (no embedding span)."""
        span_names: list[str] = []

        from contextlib import contextmanager

        @contextmanager
        def _capture_span(name, attributes=None, openinference_span_kind=None):
            span_names.append(name)
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            yield mock_span

        with (
            patch.object(retriever_module, "_load_config", return_value=KEYWORD_CONFIG),
            patch.object(ks_module, "keyword_search", return_value=[_make_raw_result()]),
            patch.object(retriever_module, "trace_span", side_effect=_capture_span),
        ):
            retriever_module.retrieve("Art. 3", top_k=5, search_strategy="keyword")

        assert "keyword_search" in span_names
        assert "query_embedding" not in span_names

    def test_hybrid_strategy_creates_embedding_and_hybrid_spans(self) -> None:
        """Hybrid strategy creates query_embedding + hybrid_search spans."""
        mock_embed = MagicMock(return_value=[0.0] * 1024)
        span_names: list[str] = []

        from contextlib import contextmanager

        @contextmanager
        def _capture_span(name, attributes=None, openinference_span_kind=None):
            span_names.append(name)
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            yield mock_span

        with (
            patch.object(retriever_module, "_load_config", return_value=HYBRID_CONFIG),
            patch.object(hs_module, "hybrid_search", return_value=[_make_raw_result()]),
            patch.dict("sys.modules", {"src.retrieval.query_embedding": MagicMock(embed_query=mock_embed)}),
            patch.object(retriever_module, "trace_span", side_effect=_capture_span),
        ):
            retriever_module.retrieve("chave Pix", top_k=5, search_strategy="hybrid")

        assert "query_embedding" in span_names
        assert "hybrid_search" in span_names

    def test_reranking_creates_reranking_span(self) -> None:
        """When reranking enabled, a 'reranking' span is created."""
        raw = [_make_raw_result("c1", 0.8), _make_raw_result("c2", 0.7), _make_raw_result("c3", 0.6)]
        reranked = [
            RetrievalResult(text="text c1", chunk_id="c1", document_id="doc", page_number=1, section_title=None, similarity_score=0.95),
            RetrievalResult(text="text c2", chunk_id="c2", document_id="doc", page_number=1, section_title=None, similarity_score=0.85),
        ]
        span_names: list[str] = []

        from contextlib import contextmanager

        @contextmanager
        def _capture_span(name, attributes=None, openinference_span_kind=None):
            span_names.append(name)
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            yield mock_span

        with (
            patch.object(retriever_module, "_load_config", return_value=RERANK_CONFIG),
            patch.object(ks_module, "keyword_search", return_value=raw),
            patch.object(retriever_module, "trace_span", side_effect=_capture_span),
            patch("src.retrieval.reranker.rerank", return_value=reranked),
        ):
            results = retriever_module.retrieve("test", top_k=2, search_strategy="keyword")

        assert "reranking" in span_names
        assert len(results) == 2

    def test_span_attributes_include_strategy_and_latency(self) -> None:
        """Search spans receive strategy and latency attributes."""
        captured_attrs: dict = {}

        from contextlib import contextmanager

        @contextmanager
        def _capture_span(name, attributes=None, openinference_span_kind=None):
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True

            def _set_attr(key, val):
                captured_attrs[key] = val

            mock_span.set_attribute.side_effect = _set_attr
            if name == "keyword_search":
                yield mock_span
            else:
                yield mock_span

        with (
            patch.object(retriever_module, "_load_config", return_value=KEYWORD_CONFIG),
            patch.object(ks_module, "keyword_search", return_value=[_make_raw_result()]),
            patch.object(retriever_module, "trace_span", side_effect=_capture_span),
        ):
            retriever_module.retrieve("test", top_k=5, search_strategy="keyword")

        assert "retrieval.result_count" in captured_attrs
        assert "retrieval.latency_ms" in captured_attrs

    def test_result_attributes_set_on_search_span(self) -> None:
        """Per-document attributes are set on search spans."""
        captured_attrs: dict = {}

        from contextlib import contextmanager

        @contextmanager
        def _capture_span(name, attributes=None, openinference_span_kind=None):
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True

            def _set_attr(key, val):
                captured_attrs[key] = val

            mock_span.set_attribute.side_effect = _set_attr
            yield mock_span

        mock_embed = MagicMock(return_value=[0.0] * 1024)
        with (
            patch.object(retriever_module, "_load_config", return_value=VECTOR_CONFIG),
            patch.object(vs_module, "vector_search", return_value=[_make_raw_result("c1", 0.9)]),
            patch.dict("sys.modules", {"src.retrieval.query_embedding": MagicMock(embed_query=mock_embed)}),
            patch.object(retriever_module, "trace_span", side_effect=_capture_span),
        ):
            retriever_module.retrieve("test", top_k=5, search_strategy="vector")

        assert "retrieval.documents.0.document.id" in captured_attrs
        assert captured_attrs["retrieval.documents.0.document.id"] == "c1"
        assert "retrieval.documents.0.document.score" in captured_attrs


# --- Issue 2: Enriched span attributes on RAG pipeline ---


class TestRAGPipelineEnrichedSpans:
    """Verify enriched attributes on RAG pipeline spans."""

    def _make_chunks(self) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                text="Chunk text about Pix",
                chunk_id="c1",
                document_id="manual_pix",
                page_number=10,
                section_title="Art. 1",
                similarity_score=0.85,
            ),
            RetrievalResult(
                text="Another chunk about regulation",
                chunk_id="c2",
                document_id="manual_pix",
                page_number=15,
                section_title="Art. 2",
                similarity_score=0.72,
            ),
        ]

    def test_rag_parent_span_has_config_attributes(self) -> None:
        """rag_pipeline span captures top_k, max_chunks, max_context_tokens."""
        captured: dict[str, dict] = {}

        from contextlib import contextmanager

        @contextmanager
        def _capture_span(name, attributes=None, openinference_span_kind=None):
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            attrs = {}
            mock_span.set_attribute.side_effect = lambda k, v: attrs.update({k: v})
            captured[name] = attrs
            yield mock_span

        chunks = self._make_chunks()
        mock_llm = MagicMock()
        mock_llm.model = "llama3.2:3b"
        mock_llm.generate.return_value = ("Answer", MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150))

        with (
            patch("src.rag.rag_pipeline.trace_span", side_effect=_capture_span),
            patch("src.rag.rag_pipeline.span_set_input"),
            patch("src.rag.rag_pipeline.span_set_output"),
            patch("src.rag.rag_pipeline.retrieve", return_value=chunks),
            patch("src.rag.rag_pipeline.build_context", return_value="Context text"),
            patch("src.rag.rag_pipeline.build_prompt", return_value="Full prompt"),
        ):
            from src.rag.rag_pipeline import answer_query
            answer_query("test query", llm=mock_llm, top_k=5, max_context_tokens=4096)

        parent_attrs = captured.get("rag_pipeline", {})
        assert parent_attrs.get("rag.top_k") == 5
        assert parent_attrs.get("rag.max_context_tokens") == 4096

    def test_retrieval_span_has_score_statistics(self) -> None:
        """Retrieval span captures score.max, score.min, score.mean."""
        captured: dict[str, dict] = {}

        from contextlib import contextmanager

        @contextmanager
        def _capture_span(name, attributes=None, openinference_span_kind=None):
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            attrs = {}
            mock_span.set_attribute.side_effect = lambda k, v: attrs.update({k: v})
            captured[name] = attrs
            yield mock_span

        chunks = self._make_chunks()
        mock_llm = MagicMock()
        mock_llm.model = "llama3.2:3b"
        mock_llm.generate.return_value = ("Answer", MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150))

        with (
            patch("src.rag.rag_pipeline.trace_span", side_effect=_capture_span),
            patch("src.rag.rag_pipeline.span_set_input"),
            patch("src.rag.rag_pipeline.span_set_output"),
            patch("src.rag.rag_pipeline.retrieve", return_value=chunks),
            patch("src.rag.rag_pipeline.build_context", return_value="Context"),
            patch("src.rag.rag_pipeline.build_prompt", return_value="Prompt"),
        ):
            from src.rag.rag_pipeline import answer_query
            answer_query("test", llm=mock_llm, top_k=5, max_context_tokens=2048)

        ret_attrs = captured.get("retrieval", {})
        assert ret_attrs.get("retrieval.result_count") == 2
        assert ret_attrs.get("retrieval.unique_documents") == 1
        assert ret_attrs.get("retrieval.score.max") == 0.85
        assert ret_attrs.get("retrieval.score.min") == 0.72
        assert abs(ret_attrs.get("retrieval.score.mean", 0) - 0.785) < 0.001

    def test_llm_span_has_model_config_and_response_length(self) -> None:
        """LLM span captures model config params and response length."""
        captured: dict[str, dict] = {}

        from contextlib import contextmanager

        @contextmanager
        def _capture_span(name, attributes=None, openinference_span_kind=None):
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            attrs = {}
            mock_span.set_attribute.side_effect = lambda k, v: attrs.update({k: v})
            captured[name] = attrs
            yield mock_span

        chunks = self._make_chunks()
        mock_llm = MagicMock()
        mock_llm.model = "llama3.2:3b"
        mock_llm.temperature = 0
        mock_llm.num_ctx = 4096
        mock_llm.num_predict = 1024
        mock_llm.generate.return_value = ("Resposta sobre Pix", MagicMock(prompt_tokens=200, completion_tokens=80, total_tokens=280))

        with (
            patch("src.rag.rag_pipeline.trace_span", side_effect=_capture_span),
            patch("src.rag.rag_pipeline.span_set_input"),
            patch("src.rag.rag_pipeline.span_set_output"),
            patch("src.rag.rag_pipeline.retrieve", return_value=chunks),
            patch("src.rag.rag_pipeline.build_context", return_value="Context"),
            patch("src.rag.rag_pipeline.build_prompt", return_value="Prompt"),
        ):
            from src.rag.rag_pipeline import answer_query
            answer_query("test", llm=mock_llm, top_k=5, max_context_tokens=2048)

        llm_attrs = captured.get("llm_generation", {})
        assert llm_attrs.get("llm.model_name") == "llama3.2:3b"
        assert llm_attrs.get("llm.num_ctx") == 4096
        assert llm_attrs.get("llm.num_predict") == 1024
        assert llm_attrs.get("llm.temperature") == 0
        assert llm_attrs.get("llm.response_length") == len("Resposta sobre Pix")
        assert llm_attrs.get("llm.token_count.total") == 280

    def test_context_building_span_enriched(self) -> None:
        """Context building span captures char_count and chunk_count."""
        captured: dict[str, dict] = {}

        from contextlib import contextmanager

        @contextmanager
        def _capture_span(name, attributes=None, openinference_span_kind=None):
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            attrs = {}
            mock_span.set_attribute.side_effect = lambda k, v: attrs.update({k: v})
            captured[name] = attrs
            yield mock_span

        chunks = self._make_chunks()
        mock_llm = MagicMock()
        mock_llm.model = "test"
        mock_llm.generate.return_value = ("ans", MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15))

        context_text = "Built context with regulatory text"

        with (
            patch("src.rag.rag_pipeline.trace_span", side_effect=_capture_span),
            patch("src.rag.rag_pipeline.span_set_input"),
            patch("src.rag.rag_pipeline.span_set_output"),
            patch("src.rag.rag_pipeline.retrieve", return_value=chunks),
            patch("src.rag.rag_pipeline.build_context", return_value=context_text),
            patch("src.rag.rag_pipeline.build_prompt", return_value="Prompt"),
        ):
            from src.rag.rag_pipeline import answer_query
            answer_query("test", llm=mock_llm, top_k=5, max_context_tokens=2048)

        ctx_attrs = captured.get("context_building", {})
        assert ctx_attrs.get("context.char_count") == len(context_text)
        assert ctx_attrs.get("context.chunk_count") == 2


# --- Tracing fallback (no-op when unavailable) ---


class TestTracingFallbackSafety:
    """Verify that retriever works fine when tracing is unavailable."""

    def test_retrieve_works_without_tracing(self) -> None:
        """retrieve() succeeds when trace_span yields None (no-op)."""
        from contextlib import contextmanager

        @contextmanager
        def _noop_span(name, attributes=None, openinference_span_kind=None):
            yield None

        with (
            patch.object(retriever_module, "_load_config", return_value=KEYWORD_CONFIG),
            patch.object(ks_module, "keyword_search", return_value=[_make_raw_result()]),
            patch.object(retriever_module, "trace_span", side_effect=_noop_span),
            patch.object(retriever_module, "span_set_input"),
            patch.object(retriever_module, "span_set_output"),
        ):
            results = retriever_module.retrieve("test", top_k=5, search_strategy="keyword")
            assert len(results) == 1
