"""Dataset loading for chunk corpus."""

import json
from pathlib import Path
from typing import Iterator

from .models import Chunk


def load_chunks_jsonl(path: Path) -> Iterator[Chunk]:
    """
    Load chunk dataset from JSONL file.

    Yields Chunk objects one at a time. Memory-efficient for large corpora.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            yield Chunk.model_validate(record)
