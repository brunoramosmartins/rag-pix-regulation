"""Unit tests for text normalization utilities."""

from src.ingestion.text_cleaner import (
    normalize_whitespace,
    fix_line_breaks,
    clean_text,
)


def test_normalize_whitespace() -> None:
    """normalize_whitespace should collapse multiple spaces and tabs."""
    text = "Hello    world\t\t!"
    result = normalize_whitespace(text)

    assert result == "Hello world !"


def test_fix_line_breaks() -> None:
    """fix_line_breaks should remove artificial line breaks."""
    text = "This is a\nbroken sentence."
    result = fix_line_breaks(text)

    assert result == "This is a broken sentence."


def test_preserve_paragraphs() -> None:
    """fix_line_breaks should preserve paragraph separation."""
    text = "Paragraph one.\n\nParagraph two."
    result = fix_line_breaks(text)

    assert result == "Paragraph one.\n\nParagraph two."


def test_clean_text_pipeline() -> None:
    """clean_text should apply the full normalization pipeline."""
    text = "Hello    world\nthis is\nbroken."

    result = clean_text(text)

    assert "Hello world this is broken." == result


def test_preserve_legal_markers() -> None:
    """clean_text should preserve legal structure markers (e.g., Art., §)."""
    text = "Art. 2º\n§1º This paragraph defines additional conditions."

    result = clean_text(text)

    assert "Art. 2º" in result
    assert "§1º" in result