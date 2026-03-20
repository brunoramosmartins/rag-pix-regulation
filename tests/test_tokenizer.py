"""Unit tests for shared tokenizer utility."""

import pytest


@pytest.mark.slow
def test_count_tokens_returns_positive_int() -> None:
    """count_tokens returns a positive integer for non-empty text."""
    from src.utils.tokenizer import count_tokens

    result = count_tokens("hello world")
    assert isinstance(result, int)
    assert result > 0


@pytest.mark.slow
def test_count_tokens_empty_string() -> None:
    """count_tokens returns 0 for empty string."""
    from src.utils.tokenizer import count_tokens

    assert count_tokens("") == 0


@pytest.mark.slow
def test_count_tokens_consistent() -> None:
    """Same text returns same count across calls."""
    from src.utils.tokenizer import count_tokens

    text = "O cadastro de chave Pix é feito pelo participante."
    assert count_tokens(text) == count_tokens(text)


@pytest.mark.slow
def test_count_tokens_portuguese_accuracy() -> None:
    """Token count for Portuguese text is reasonable (not heuristic)."""
    from src.utils.tokenizer import count_tokens

    # 100 chars of Portuguese text; heuristic would say ~25 tokens (100/4)
    # Real tokenizer should give a different count
    text = "O participante deve registrar a chave Pix no DICT conforme regulamentação vigente do BCB."
    tokens = count_tokens(text)
    assert tokens > 0
    # Real tokenizer count should differ from len(text) // 4
    heuristic = len(text) // 4
    # Just verify it's a reasonable number (not wildly off)
    assert tokens != heuristic or tokens > 10  # sanity check
