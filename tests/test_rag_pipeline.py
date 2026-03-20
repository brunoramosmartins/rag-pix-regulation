"""Unit tests for RAG pipeline."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.llm import LLMClient
from src.rag.rag_pipeline import RAGResponse, answer_query
from src.retrieval.retriever import RetrievalResult

# Import prompt_template directly to avoid pulling in rag_pipeline (sentence_transformers)
_prompt_path = (
    Path(__file__).resolve().parent.parent / "src" / "rag" / "prompt_template.py"
)

_spec = importlib.util.spec_from_file_location("prompt_template", _prompt_path)
_prompt_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompt_mod)

SYSTEM_INSTRUCTION = _prompt_mod.SYSTEM_INSTRUCTION
build_prompt = _prompt_mod.build_prompt


class MockLLM(LLMClient):
    """Mock LLM for testing orchestration."""

    def generate(self, prompt: str):
        return "test answer", None


def _chunk(text: str, doc: str = "manual_dict", page: int = 1) -> RetrievalResult:
    return RetrievalResult(
        text=text,
        chunk_id=f"{doc}_p{page}_s0_c0",
        document_id=doc,
        page_number=page,
        section_title=None,
        similarity_score=0.9,
    )


def _mock_retrieve(query: str, top_k: int) -> list[RetrievalResult]:
    return [_chunk("Regulatory content.", doc="manual_dict", page=5)]


def test_system_instruction_is_robust() -> None:
    """System instruction includes key rules for regulatory RAG."""
    assert "ONLY" in SYSTEM_INSTRUCTION or "exclusively" in SYSTEM_INSTRUCTION
    assert "não está disponível" in SYSTEM_INSTRUCTION
    assert "cite" in SYSTEM_INSTRUCTION.lower()
    assert "context" in SYSTEM_INSTRUCTION.lower()


def test_build_prompt_includes_sections() -> None:
    """build_prompt produces structured prompt with delimiters."""
    prompt = build_prompt("Contexto aqui.", "Pergunta?")
    assert "System Instruction" in prompt
    assert "Regulatory Context" in prompt
    assert "User Question" in prompt
    assert "Contexto aqui." in prompt
    assert "Pergunta?" in prompt


def test_answer_query_returns_rag_response() -> None:
    """answer_query returns RAGResponse with all fields."""
    response = answer_query(
        "Como cadastrar chave?",
        llm=MockLLM(),
        max_context_tokens=1500,
        retriever=_mock_retrieve,
    )

    assert isinstance(response, RAGResponse)
    assert response.query == "Como cadastrar chave?"
    assert "test answer" in response.answer
    assert response.context
    assert len(response.retrieved_chunks) == 1
    assert response.citations == ["manual_dict, p. 5"]


def test_answer_query_citations_deterministic() -> None:
    """Citations come from retrieved chunks, not LLM."""

    def mock_retrieve(q: str, top_k: int) -> list[RetrievalResult]:
        return [
            _chunk("A", doc="d1", page=1),
            _chunk("B", doc="d1", page=2),
        ]

    response = answer_query(
        "q", llm=MockLLM(), max_context_tokens=1500, retriever=mock_retrieve,
    )

    assert "d1, p. 1" in response.citations
    assert "d1, p. 2" in response.citations


def test_answer_query_passes_top_k() -> None:
    """answer_query passes top_k to retriever."""
    calls: list[tuple[str, int]] = []

    def mock_retrieve(query: str, top_k: int) -> list[RetrievalResult]:
        calls.append((query, top_k))
        return []

    answer_query(
        "q", llm=MockLLM(), top_k=7, max_context_tokens=1500, retriever=mock_retrieve,
    )

    assert calls == [("q", 7)]


@pytest.mark.skipif(
    True,
    reason="Requires sentence_transformers/Weaviate; run manually when env is ready",
)
def test_answer_query_integration() -> None:
    """Integration test with citation footer."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = ("Resposta do modelo.", None)

    def mock_retrieve(query: str, top_k: int):
        return [RetrievalResult("chunk1", "c1", "manual_dict", 1, None, 0.9)]

    response = answer_query("Qual o limite Pix?", llm=mock_llm, retriever=mock_retrieve)

    assert isinstance(response, RAGResponse)
    assert "Resposta do modelo" in response.answer
    assert "Fontes consultadas" in response.answer
    assert response.citations == ["manual_dict p.1"]