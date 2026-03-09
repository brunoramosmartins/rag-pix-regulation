"""Unit tests for RAG pipeline."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Import prompt_template directly to avoid pulling in rag_pipeline (sentence_transformers)
_prompt_path = Path(__file__).resolve().parent.parent / "src" / "rag" / "prompt_template.py"
_spec = importlib.util.spec_from_file_location("prompt_template", _prompt_path)
_prompt_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompt_mod)
SYSTEM_INSTRUCTION = _prompt_mod.SYSTEM_INSTRUCTION
build_prompt = _prompt_mod.build_prompt


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


@pytest.mark.skipif(
    True,  # Skip when heavy deps fail; set to False to run full pipeline tests
    reason="Requires sentence_transformers/Weaviate; run manually when env is ready",
)
def test_answer_query_integration() -> None:
    """answer_query returns RAGResponse with citation footer (integration test)."""
    from src.rag.rag_pipeline import RAGResponse, answer_query
    from src.retrieval.retriever import RetrievalResult

    mock_llm = MagicMock()
    mock_llm.generate.return_value = ("Resposta do modelo.", None)

    def mock_retrieve(query: str, top_k: int):
        return [RetrievalResult("chunk1", "c1", "manual_dict", 1, None, 0.9)]

    response = answer_query("Qual o limite Pix?", llm=mock_llm, retriever=mock_retrieve)

    assert isinstance(response, RAGResponse)
    assert "Resposta do modelo" in response.answer
    assert "Fontes consultadas" in response.answer
    assert response.citations == ["manual_dict p.1"]
