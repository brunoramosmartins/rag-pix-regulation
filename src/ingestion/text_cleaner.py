"""Text normalization utilities for regulatory documents."""

import re

# Footer pattern for DICT MED: "N • Manual Operacional do Diretório de Identificadores de Contas Transacionais (DICT) – Versão 8.1"
_DICT_FOOTER_PATTERN = re.compile(
    r"\d+\s*[•·]\s*Manual Operacional do Diretório de Identificadores de Contas Transacionais\s*\(DICT\)\s*[–\-]\s*Versão\s*[\d.]+",
    re.IGNORECASE | re.MULTILINE,
)


def remove_repeated_footer(text: str, pattern: re.Pattern | None = None) -> str:
    """
    Remove repeated footer lines (e.g., document title + page number).

    Default pattern matches DICT MED format:
    "N • Manual Operacional do Diretório de Identificadores de Contas Transacionais (DICT) – Versão X.X"
    """
    pattern = pattern or _DICT_FOOTER_PATTERN
    text = pattern.sub("", text)
    return text.strip()


def preserve_legal_markers(text: str) -> str:
    """
    Ensure legal markers (Art., §, numbered sections) start on a new line
    so they remain identifiable after line-break normalization.

    Uses double newlines so markers survive fix_line_breaks, which collapses
    single newlines. Essential for chunking and citation in RAG pipelines.
    """
    # Art. 1º, Art. 2º, etc.
    text = re.sub(r"\s*(Art\.\s*\d+º?)", r"\n\n\1", text)
    # §1º, §2º, etc.
    text = re.sub(r"\s*(§\s*\d+º?)", r"\n\n\1", text)
    # Section numbers: 3.1, 4.2.1, etc.
    text = re.sub(r"\s*(\d+\.\d+(?:\.\d+)*)\s+", r"\n\n\1 ", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph structure."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fix_line_breaks(text: str) -> str:
    """Fix artificial line breaks introduced by PDF extraction."""
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text


def validate_text(text: str) -> bool:
    """
    Validate that cleaned text meets basic consistency requirements.

    Returns False if text is empty, too short, or contains invalid sequences.
    """
    if not text or not text.strip():
        return False
    # Reject if mostly non-printable or replacement chars
    printable_ratio = sum(1 for c in text if c.isprintable() or c in "\n\t") / max(
        len(text), 1
    )
    return printable_ratio >= 0.9


def clean_text(text: str, remove_footer: bool = True) -> str:
    """Apply all text normalization steps.

    Order is important:
    1. preserve structural markers (Art., §, section numbers)
    2. collapse artificial line breaks
    3. normalize whitespace
    """
    if remove_footer:
        text = remove_repeated_footer(text)
    text = preserve_legal_markers(text)
    text = fix_line_breaks(text)
    text = normalize_whitespace(text)
    return text.strip()
