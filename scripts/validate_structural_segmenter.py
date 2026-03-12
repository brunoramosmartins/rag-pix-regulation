#!/usr/bin/env python
"""
Validate structural segmentation on real corpus.

Run from project root:

    python scripts/validate_structural_segmenter.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import segment_records  # noqa: E402


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

    # Segment first 5 pages
    sample = records[:5]
    segments = segment_records(sample)

    print(f"Total records: {len(records)}")
    print(f"Sample: {len(sample)} pages -> {len(segments)} segments\n")

    for i, seg in enumerate(segments[:10]):
        print(
            f"--- Segment {i} (page {seg['page_number']}, idx {seg['segment_index']}) ---"
        )
        print(f"section_title: {seg.get('section_title')}")
        print(f"article_numbers: {seg.get('article_numbers')}")
        print(f"text (first 200 chars): {seg['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()
