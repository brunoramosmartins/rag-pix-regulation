"""Text normalization utilities for regulatory documents."""

import re


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph structure."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fix_line_breaks(text: str) -> str:
    """Fiz artificial line breaks introduced by PDF extraction."""
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text


def clean_text(text: str) -> str:
    """Apply all text normalization steps."""
    text = normalize_whitespace(text)
    text = fix_line_breaks(text)
    return text