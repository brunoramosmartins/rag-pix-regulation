# Embedding and Indexing Pipeline

Chunks → vectors → Weaviate.

```mermaid
flowchart LR
    subgraph Input
        CHUNKS[corpus_chunks.jsonl]
    end

    subgraph Processing
        EMB[Embedding Generator<br/>BGE-M3]
        IDX[Weaviate Indexer]
    end

    subgraph Storage
        WV[Weaviate<br/>Vector DB]
    end

    CHUNKS --> EMB
    EMB --> IDX
    IDX --> WV
```

## Scripts

```bash
python scripts/init_weaviate.py   # Create collection schema
python scripts/run_indexing.py    # Embed and index chunks
```

## Single Command

```bash
python scripts/run_pipeline.py    # Full pipeline: ingestion → chunking → init → indexing
```
