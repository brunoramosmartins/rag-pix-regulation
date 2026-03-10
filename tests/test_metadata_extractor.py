"""Unit tests for metadata extraction."""

from src.ingestion.metadata_extractor import (
    enrich_page_metadata,
    extract_article_markers,
    extract_section_title,
)


def test_extract_section_title_simple() -> None:
    """extract_section_title detects '1 Chaves Pix' pattern."""
    text = "1 Chaves Pix\n\nAs chaves Pix serão armazenadas..."
    assert extract_section_title(text) == "1 Chaves Pix"


def test_extract_section_title_numbered() -> None:
    """extract_section_title detects '3.1 Fluxo de registro' pattern."""
    text = "3.1 Fluxo de registro de chave por solicitação do usuário final"
    assert (
        extract_section_title(text)
        == "3.1 Fluxo de registro de chave por solicitação do usuário final"
    )


def test_extract_section_title_none() -> None:
    """extract_section_title returns None when no section found."""
    text = "Conteúdo sem título de seção no início."
    assert extract_section_title(text) is None


def test_extract_section_title_after_content() -> None:
    """extract_section_title finds section at start of a line."""
    text = "Prefácio.\n\n4 Fluxo de exclusão de chave\n\nO fluxo define..."
    assert extract_section_title(text) == "4 Fluxo de exclusão de chave"


def test_extract_section_title_rejects_numbered_step() -> None:
    """extract_section_title rejects numbered process steps (false positive)."""
    text = "3 PSP do usuário final recebe comunicação do DICT."
    assert extract_section_title(text) is None


def test_extract_section_title_rejects_flow_table_row() -> None:
    """extract_section_title rejects flow table rows (Camada, Tipo, Descrição)."""
    text = "# Camada Tipo Descrição"
    assert extract_section_title(text) is None


def test_extract_section_title_rejects_long_line() -> None:
    """extract_section_title rejects lines with more than 12 words."""
    text = "1 " + " ".join(["word"] * 15)
    assert extract_section_title(text) is None


def test_extract_section_title_rejects_lowercase_first_word() -> None:
    """extract_section_title rejects when first content word is lowercase."""
    text = "3 usuário final envia comunicação ao PSP"
    assert extract_section_title(text) is None


def test_extract_section_first_match() -> None:
    """extract_section_title returns first valid section when multiple exist."""
    text = "1 Introdução\n\n2 Chaves Pix"
    assert extract_section_title(text) == "1 Introdução"


def test_extract_article_markers_art() -> None:
    """extract_article_markers finds Art. Xº patterns."""
    text = "Art. 1º A chave Pix será armazenada. Art. 2º O usuário deve validar."
    result = extract_article_markers(text)
    assert "Art. 1º" in result
    assert "Art. 2º" in result


def test_extract_article_markers_paragraph() -> None:
    """extract_article_markers finds §Xº patterns."""
    text = "§1º O usuário deve validar. §2º A validação é obrigatória."
    result = extract_article_markers(text)
    assert "§1º" in result
    assert "§2º" in result


def test_extract_article_markers_mixed() -> None:
    """extract_article_markers finds both Art. and § in order."""
    text = "Art. 3º Regras gerais. §1º Primeiro parágrafo. §2º Segundo parágrafo."
    result = extract_article_markers(text)
    assert result == ["Art. 3º", "§1º", "§2º"]


def test_extract_article_markers_none() -> None:
    """extract_article_markers returns empty list when none found."""
    text = "Texto sem marcadores de artigo."
    assert extract_article_markers(text) == []


def test_enrich_page_metadata() -> None:
    """enrich_page_metadata returns section_title and article_numbers."""
    text = "3.1 Fluxo de registro\n\nArt. 1º A chave deve ser validada. §1º O PSP verifica."
    section_title, article_numbers = enrich_page_metadata(text)
    assert section_title == "3.1 Fluxo de registro"
    assert "Art. 1º" in article_numbers
    assert "§1º" in article_numbers


def test_enrich_page_metadata_fallback() -> None:
    """enrich_page_metadata returns None and [] when no metadata found."""
    text = "Conteúdo introdutório sem estrutura formal."
    section_title, article_numbers = enrich_page_metadata(text)
    assert section_title is None
    assert article_numbers == []
