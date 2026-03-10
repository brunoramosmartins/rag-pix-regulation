"""RAG module - Retrieval-Augmented Generation pipeline orchestration."""

from .context_builder import build_context
from .prompt_template import build_prompt
from .rag_pipeline import RAGResponse, answer_query

__all__ = [
    "RAGResponse",
    "answer_query",
    "build_context",
    "build_prompt",
]