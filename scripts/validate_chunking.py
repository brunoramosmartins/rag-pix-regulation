#!/usr/bin/env python
"""
Validate token chunking on real corpus.

Run from project root:

    python scripts/run_ingestion.py
    python scripts/validate_structural_segmenter.py  # optional: segment first
    python scripts/validate_chunking.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import segment_records, chunk_records  # noqa: E402


def main() -> None:
    corpus_path = PROJECT_ROOT / "data" / "processed" / "corpus_pages.jsonl"
    if not corpus_path.exists():
        print(
            "ERROR: corpus_pages.jsonl not found. Run: python scripts/run_ingestion.py"
        )
        sys.exit(1)

    records = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    segments = segment_records(records[:3])
    chunks = chunk_records(segments, chunk_size=500, chunk_overlap=50)

    print(
        f"Pages: {len(records[:3])} -> Segments: {len(segments)} -> Chunks: {len(chunks)}\n"
    )

    for i, chunk in enumerate(chunks[:8]):
        print(f"--- Chunk {i} ---")
        print(f"chunk_id: {chunk['chunk_id']}")
        print(f"token_count: {chunk['token_count']}")
        print(f"text (first 200 chars): {chunk['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()
