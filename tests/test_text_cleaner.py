"""Unit tests for text normalization utilities."""

from src.ingestion.text_cleaner import (
    clean_text,
    fix_line_breaks,
    normalize_whitespace,
    preserve_legal_markers,
    remove_repeated_footer,
    validate_text,
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


def test_remove_repeated_footer() -> None:
    """remove_repeated_footer should remove DICT MED footer pattern."""
    text = "Conteúdo da página.\n\n5 • Manual Operacional do Diretório de Identificadores de Contas Transacionais (DICT) – Versão 8.1"
    result = remove_repeated_footer(text)
    assert "5 • Manual Operacional" not in result
    assert "Conteúdo da página." in result


def test_remove_repeated_footer_preserves_content() -> None:
    """remove_repeated_footer should not remove similar-looking content."""
    text = "Manual Operacional do Diretório de Identificadores de Contas Transacionais (DICT) no contexto do documento."
    result = remove_repeated_footer(text)
    assert "Manual Operacional" in result


def test_validate_text_valid() -> None:
    """validate_text returns True for readable content."""
    assert validate_text("Art. 1º O Pix é um meio de pagamento.") is True


def test_preserve_marker_existing_line() -> None:
    """clean_text preserves markers already on their own line without duplicating."""
    text = "Art. 1º\nTexto."
    result = clean_text(text)
    assert "Art. 1º" in result
    assert result.count("Art. 1º") == 1


def test_validate_text_empty() -> None:
    """validate_text returns False for empty or whitespace-only text."""
    assert validate_text("") is False
    assert validate_text("   \n\t  ") is False


def test_preserve_legal_markers_art() -> None:
    """preserve_legal_markers ensures Art. Xº starts on a new line."""
    text = "Conclusão. Art. 1º A chave Pix será armazenada."
    result = preserve_legal_markers(text)
    assert "Art. 1º" in result
    assert result.index("Art. 1º") == 0 or result[result.index("Art. 1º") - 1] == "\n"


def test_preserve_legal_markers_paragraph() -> None:
    """preserve_legal_markers ensures §Xº starts on a new line."""
    text = "Conforme acima. §1º O usuário deve validar."
    result = preserve_legal_markers(text)
    assert "§1º" in result
    assert (
        result.index("§1º") == 0
        or result[result.index("§1º") - 2 : result.index("§1º")] == "\n\n"
    )


def test_preserve_legal_markers_survives_pipeline() -> None:
    """Legal markers remain on separate lines after full clean_text pipeline."""
    text = "Texto introdutório. Art. 2º A chave deve ser validada."
    result = clean_text(text)
    assert "Art. 2º" in result
    # Art. 2º should start its own line (paragraph boundary preserved)
    lines = result.split("\n")
    assert any("Art. 2º" in line for line in lines)
