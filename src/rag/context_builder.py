"""Context builder with token-based truncation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrieval.models import RetrievalResult

# Approximate tokens per char for Portuguese (conservative)
CHARS_PER_TOKEN = 4


def _count_tokens(text: str) -> int:
    """
    Estimate token count for truncation.

    Uses character-based heuristic (1 token ≈ 4 chars) to avoid
    loading a heavy tokenizer. Sufficient for context size safety.
    """
    return len(text) // CHARS_PER_TOKEN


def build_context(
    chunks: list["RetrievalResult"],
    max_chunks: int = 5,
    max_tokens: int | None = None,
) -> str:
    """
    Build context string from retrieved chunks with truncation.

    Parameters
    ----------
    chunks : list[RetrievalResult]
        Retrieved chunks (ordered by relevance).
    max_chunks : int, default 5
        Maximum number of chunks to include.
    max_tokens : int | None
        Maximum total tokens. If exceeded, truncate oldest chunks.
        If None, no token limit.

    Returns
    -------
    str
        Concatenated context with source markers.
    """
    limited = chunks[:max_chunks]
    parts: list[str] = []

    for r in limited:
        marker = f"[{r.document_id} p.{r.page_number}]"
        parts.append(f"{marker}\n{r.text}")

    context = "\n\n".join(parts)

    if max_tokens is None:
        return context

    current_tokens = _count_tokens(context)
    if current_tokens <= max_tokens:
        return context

    # Truncate from the end (oldest chunks) until under limit
    for i in range(len(limited) - 1, 0, -1):
        truncated_parts = parts[:i]
        truncated_context = "\n\n".join(truncated_parts)
        if _count_tokens(truncated_context) <= max_tokens:
            return truncated_context

    # Fallback: truncate last chunk text to fit
    if parts:
        max_chars = max_tokens * CHARS_PER_TOKEN
        prefix = "\n\n".join(parts[:-1])
        separator = "\n\n" if prefix else ""
        allowed_last = max_chars - len(prefix) - len(separator)
        if allowed_last > 0:
            parts[-1] = parts[-1][:allowed_last].rstrip() + "..."
    return "\n\n".join(parts)
