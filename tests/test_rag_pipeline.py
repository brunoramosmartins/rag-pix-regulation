"""Unit tests for RAG pipeline (mocked LLM and retriever)."""

from src.llm import LLMClient
from src.rag.rag_pipeline import RAGResponse, answer_query
from src.retrieval.models import RetrievalResult


class MockLLM(LLMClient):
    """Mock LLM for testing orchestration."""

    def generate(self, prompt: str) -> str:
        return "test answer"


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


def test_answer_query_returns_rag_response() -> None:
    """answer_query returns RAGResponse with all fields."""
    response = answer_query(
        "Como cadastrar chave?",
        llm=MockLLM(),
        retriever=_mock_retrieve,
    )
    assert isinstance(response, RAGResponse)
    assert response.query == "Como cadastrar chave?"
    assert response.answer == "test answer"
    assert response.context
    assert len(response.retrieved_chunks) == 1
    assert response.citations == ["manual_dict p.5"]


def test_answer_query_citations_deterministic() -> None:
    """Citations come from retrieved chunks, not LLM."""
    def mock_retrieve(q: str, top_k: int) -> list[RetrievalResult]:
        return [
            _chunk("A", doc="d1", page=1),
            _chunk("B", doc="d1", page=2),
        ]
    response = answer_query("q", llm=MockLLM(), retriever=mock_retrieve)
    assert "d1 p.1" in response.citations
    assert "d1 p.2" in response.citations


def test_answer_query_passes_top_k() -> None:
    """answer_query passes top_k to retriever."""
    calls: list[tuple[str, int]] = []

    def mock_retrieve(query: str, top_k: int) -> list[RetrievalResult]:
        calls.append((query, top_k))
        return []

    answer_query("q", llm=MockLLM(), top_k=7, retriever=mock_retrieve)
    assert calls == [("q", 7)]
