# Chunking Pipeline

Structured pages → retrieval-optimized chunks.

```mermaid
flowchart LR
    subgraph Input
        PAGES[corpus_pages.jsonl]
    end

    subgraph Chunking
        SEG[Structural Segmenter]
        TOK[Token Chunker]
    end

    subgraph Output
        CHUNKS[corpus_chunks.jsonl]
    end

    PAGES --> SEG
    SEG --> TOK
    TOK --> CHUNKS
```

## Strategy

1. **Structural segmentation** — Split by sections, articles, paragraphs
2. **Token chunking** — Respect model context limits (overlap, max tokens)

## Script

```bash
python scripts/run_chunking.py
```

## Output Format

Each chunk: `chunk_id`, `document_id`, `page_number`, `section_title`, `text`, `token_count`
