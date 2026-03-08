"""Serialization of chunk objects to structured JSONL format."""

import json
import re
from pathlib import Path
from typing import Iterable

from .models import Chunk

CHUNK_ID_PATTERN = re.compile(r"^.+_p\d+_s\d+_c\d+$")


def chunk_to_record(chunk: Chunk) -> dict:
    """
    Convert a Chunk to a flat JSONL record.

    Flat schema improves vector DB ingestion and retrieval filtering.
    """
    return chunk.model_dump()


def save_chunks_jsonl(chunks: Iterable[Chunk], path: Path) -> int:
    """
    Write chunks to JSONL file incrementally.

    Streaming write supports large corpora without loading all in memory.
    Returns the number of records written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = chunk_to_record(chunk)
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
            count += 1

    return count


def validate_chunk_record(
    record: dict,
    *,
    chunk_size: int | None = None,
    require_char_offsets: bool = False,
) -> None:
    """
    Validate a chunk record schema and consistency.

    Raises ValueError if validation fails.
    """
    required = {
        "chunk_id",
        "document_id",
        "page_number",
        "segment_index",
        "chunk_index",
        "section_title",
        "article_numbers",
        "source_file",
        "text",
        "token_count",
    }
    missing = required - set(record.keys())
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    if not record["text"] or not str(record["text"]).strip():
        raise ValueError("text must not be empty")

    if not isinstance(record["token_count"], (int, float)) or record["token_count"] < 0:
        raise ValueError("token_count must be a non-negative number")

    if chunk_size is not None and record["token_count"] > chunk_size:
        raise ValueError(
            f"token_count ({record['token_count']}) exceeds chunk_size ({chunk_size})"
        )

    chunk_id = str(record["chunk_id"])
    if not CHUNK_ID_PATTERN.match(chunk_id):
        raise ValueError(
            f"chunk_id must follow document_id_pN_sN_cN pattern, got: {chunk_id}"
        )

    if require_char_offsets and ("char_start" not in record or "char_end" not in record):
        raise ValueError("char_start and char_end required when require_char_offsets=True")


def validate_chunk_dataset(
    path: Path,
    *,
    chunk_size: int | None = None,
) -> tuple[int, list[str]]:
    """
    Validate entire chunk dataset for schema consistency.

    Returns (record_count, list of error messages).
    Empty error list means all records passed validation.
    """
    path = Path(path)
    errors: list[str] = []
    count = 0

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_chunk_record(record, chunk_size=chunk_size)
                count += 1
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON - {e}")
            except ValueError as e:
                errors.append(f"Line {i}: {e}")

    return count, errors
