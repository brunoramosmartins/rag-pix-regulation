"""Retrieval module - Similarity search and document retrieval."""

from .query_embedding import embed_query
from .retriever import RetrievalResult, retrieve
from .vector_search import vector_search

__all__ = [
    "embed_query",
    "vector_search",
    "retrieve",
    "RetrievalResult",
]
