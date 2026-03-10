"""Structural metadata extraction for regulatory documents."""

import re

# Section: "1 Chaves Pix", "3.1 Fluxo de registro", "4 Fluxo de exclusão"
_SECTION_PATTERN = re.compile(r"^\d+(?:\.\d+)*\s+.+$")

# Flow table columns in DICT manual - reject lines with 2+ of these
_FLOW_KEYWORDS = frozenset(
    {
        "camada",
        "tipo",
        "descrição",
        "ação",
        "mensagem",
        "comunicação",
    }
)

# Articles: Art. 1º, Art. 2º
_ARTICLE_PATTERN = re.compile(r"Art\.\s*\d+º?", re.IGNORECASE)

# Paragraphs: §1º, §2º
_PARAGRAPH_PATTERN = re.compile(r"§\s*\d+º?")

_MAX_SECTION_WORDS = 12


def _is_valid_section_title(candidate: str) -> bool:
    """
    Apply structural heuristics to reject false section titles.

    Rejects: numbered steps, table rows, long sentences, lines ending with period.
    """
    candidate = candidate.strip()
    if not candidate:
        return False

    # Real titles do not end with punctuation
    if candidate.endswith("."):
        return False

    words = candidate.split()
    # Real titles are short and concise
    if len(words) > _MAX_SECTION_WORDS:
        return False

    # First word after number must be capitalized
    if len(words) < 2:
        return False
    first_content_word = words[1]
    if not first_content_word or not first_content_word[0].isupper():
        return False

    # Reject flow table rows (Camada, Tipo, Descrição, etc.)
    keyword_count = sum(1 for w in words if w.lower() in _FLOW_KEYWORDS)
    if keyword_count >= 2:
        return False

    return True


def extract_section_title(text: str) -> str | None:
    """
    Extract the first valid section title from page text.

    Matches patterns like:
    - 1 Chaves Pix
    - 3.1 Fluxo de registro
    - 4 Fluxo de exclusão

    Applies heuristics to reject false positives:
    - Numbered steps (e.g. "3 PSP do usuário final recebe comunicação do DICT.")
    - Flow table rows (Camada, Tipo, Descrição)
    - Long lines (>12 words)
    - Lines ending with period

    Limits search to first 20 lines (section titles appear near top of page).
    Returns None if no valid section title is found.
    """
    for line in text.splitlines()[:20]:
        line = line.strip()
        if _SECTION_PATTERN.match(line) and _is_valid_section_title(line):
            return line
    return None


def extract_article_markers(text: str) -> list[str]:
    """
    Extract all article and paragraph markers from page text.

    Returns a list of unique markers in order of first appearance,
    e.g. ['Art. 1º', '§1º', 'Art. 2º'].
    """
    markers: list[str] = []
    seen: set[str] = set()

    for pattern in (_ARTICLE_PATTERN, _PARAGRAPH_PATTERN):
        for match in pattern.finditer(text):
            marker = match.group(0)
            if marker not in seen:
                seen.add(marker)
                markers.append(marker)

    return markers


def enrich_page_metadata(text: str) -> tuple[str | None, list[str]]:
    """
    Extract structural metadata from page text.

    Returns (section_title, article_numbers) for use when creating a Page.
    """
    section_title = extract_section_title(text)
    article_numbers = extract_article_markers(text)
    return section_title, article_numbers
