# Ingestion Pipeline

PDF documents → structured pages with metadata.

```mermaid
flowchart LR
    subgraph Input
        PDF[Regulatory PDFs<br/>data/raw/]
    end

    subgraph Ingestion
        LOAD[PDF Loader]
        PARSE[Parse & Extract]
        CLEAN[Text Cleaner]
        META[Metadata Extractor]
    end

    subgraph Output
        PAGES[corpus_pages.jsonl]
    end

    PDF --> LOAD
    LOAD --> PARSE
    PARSE --> CLEAN
    CLEAN --> META
    META --> PAGES
```

## Script

```bash
python scripts/run_ingestion.py
```

## Output Format

Each line in `corpus_pages.jsonl`:

- `document_id`, `page_number`, `text`, `metadata`
