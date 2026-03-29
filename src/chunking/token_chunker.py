"""Token-based chunking for structural segments."""

from typing import Iterable

from transformers import AutoTokenizer

from .models import Chunk, StructuralSegment, compute_content_hash

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

_tokenizer = None


def _get_tokenizer(model_name: str = DEFAULT_MODEL) -> AutoTokenizer:
    """Return tokenizer instance (cached)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer


def _tokenize_text(text: str, tokenizer: AutoTokenizer) -> list[int]:
    """Convert text to token ids."""
    return tokenizer.encode(text, add_special_tokens=False)


def _detokenize(tokens: list[int], tokenizer: AutoTokenizer) -> str:
    """Convert token ids back to string."""
    return tokenizer.decode(tokens, skip_special_tokens=True)


def _count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    """Return token count for text."""
    return len(_tokenize_text(text, tokenizer))


def _generate_chunk_id(
    document_id: str,
    page_number: int,
    segment_index: int,
    chunk_index: int,
) -> str:
    """Generate deterministic chunk identifier."""
    return f"{document_id}_p{page_number}_s{segment_index}_c{chunk_index}"


def chunk_segment(
    segment: StructuralSegment,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    tokenizer_name: str = DEFAULT_MODEL,
) -> list[Chunk]:
    """
    Split segment into token-sized chunks with overlap.

    Does not alter text content; only splits into windows.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    text = segment.text
    if not text.strip():
        return []

    tokenizer = _get_tokenizer(tokenizer_name)
    tokens = _tokenize_text(text, tokenizer)
    token_count = len(tokens)

    if token_count <= chunk_size:
        chunk_text = _detokenize(tokens, tokenizer)
        chunk_id = _generate_chunk_id(
            segment.document_id,
            segment.page_number,
            segment.segment_index,
            0,
        )
        return [
            Chunk(
                chunk_id=chunk_id,
                document_id=segment.document_id,
                page_number=segment.page_number,
                segment_index=segment.segment_index,
                chunk_index=0,
                section_title=segment.section_title,
                article_numbers=segment.article_numbers,
                source_file=segment.source_file,
                text=chunk_text,
                token_count=token_count,
                content_hash=compute_content_hash(chunk_text),
                char_start=segment.char_start,
                char_end=segment.char_end,
            )
        ]

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        window = tokens[start:end]
        chunk_text = _detokenize(window, tokenizer)
        token_window_count = len(window)

        chunk_id = _generate_chunk_id(
            segment.document_id,
            segment.page_number,
            segment.segment_index,
            chunk_index,
        )

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=segment.document_id,
                page_number=segment.page_number,
                segment_index=segment.segment_index,
                chunk_index=chunk_index,
                section_title=segment.section_title,
                article_numbers=segment.article_numbers,
                source_file=segment.source_file,
                text=chunk_text,
                token_count=token_window_count,
                content_hash=compute_content_hash(chunk_text),
                char_start=None,
                char_end=None,
            )
        )

        chunk_index += 1
        if end >= len(tokens):
            break
        start = start + chunk_size - chunk_overlap

    return chunks


def chunk_segments(
    segments: Iterable[StructuralSegment],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    tokenizer_name: str = DEFAULT_MODEL,
) -> list[Chunk]:
    """Chunk multiple segments."""
    chunks: list[Chunk] = []
    for segment in segments:
        chunks.extend(chunk_segment(segment, chunk_size, chunk_overlap, tokenizer_name))
    return chunks


def chunk_records(
    records: Iterable[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    tokenizer_name: str = DEFAULT_MODEL,
) -> list[dict]:
    """Chunk segment records (from JSONL) into chunk dicts."""
    chunks: list[dict] = []
    for rec in records:
        segment = StructuralSegment(
            document_id=rec["document_id"],
            page_number=rec["page_number"],
            section_title=rec.get("section_title"),
            article_numbers=rec.get("article_numbers", []),
            source_file=rec["source_file"],
            text=rec["text"],
            segment_index=rec["segment_index"],
            char_start=rec.get("char_start"),
            char_end=rec.get("char_end"),
        )
        for chunk in chunk_segment(segment, chunk_size, chunk_overlap, tokenizer_name):
            chunks.append(chunk.model_dump())
    return chunks
