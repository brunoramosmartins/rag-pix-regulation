"""Retriever orchestration: query embedding + vector search."""

from .models import RetrievalResult
from .query_embedding import embed_query
from .vector_search import vector_search


def retrieve(
    query: str,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """
    Retrieve top-K most relevant chunks for a query.

    Pipeline: query → embed_query → vector_search → RetrievalResult list.

    Returns
    -------
    list[RetrievalResult]
        Ranked retrieval results ordered by similarity score (highest first).
    """
    query_vector = embed_query(query)
    raw_results = vector_search(query_vector, top_k=top_k)

    return [
        RetrievalResult(
            text=r.get("text") or "",
            chunk_id=r.get("chunk_id") or "",
            document_id=r.get("document_id") or "",
            page_number=r.get("page_number") or 0,
            section_title=r.get("section_title"),
            similarity_score=r.get("similarity_score"),
            source_file=r.get("source_file"),
        )
        for r in raw_results
    ]
