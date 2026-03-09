"""Context builder with token-based truncation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrieval.retriever import RetrievalResult

CHARS_PER_TOKEN = 4


def _count_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def build_context(
    chunks: list["RetrievalResult"],
    max_chunks: int = 5,
    max_tokens: int | None = None,
) -> str:
    """
    Build context string from retrieved chunks with truncation.
    """
    limited = chunks[:max_chunks]
    parts: list[str] = []

    for r in limited:
        marker = f"[{r.document_id} p.{r.page_number}]"
        parts.append(f"{marker}\n{r.text}")

    context = "\n\n".join(parts)

    if max_tokens is None:
        return context

    if _count_tokens(context) <= max_tokens:
        return context

    for i in range(len(limited) - 1, 0, -1):
        truncated = "\n\n".join(parts[:i])
        if _count_tokens(truncated) <= max_tokens:
            return truncated

    if parts:
        max_chars = max_tokens * CHARS_PER_TOKEN
        prefix = "\n\n".join(parts[:-1])
        sep = "\n\n" if prefix else ""
        allowed = max_chars - len(prefix) - len(sep)
        if allowed > 0:
            parts[-1] = parts[-1][:allowed].rstrip() + "..."
    return "\n\n".join(parts)
