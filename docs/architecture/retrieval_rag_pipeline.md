# Retrieval and RAG Pipeline

Query → retrieval → context → prompt → LLM → response.

```mermaid
flowchart TB
    subgraph Query
        Q[User Query]
    end

    subgraph Retrieval
        E[embed_query]
        VS[vector_search]
        R[RetrievalResult list]
    end

    subgraph RAG
        CB[build_context]
        BP[build_prompt]
        LLM[LLM.generate]
    end

    subgraph Output
        ANS[Answer]
        CIT[Citations]
    end

    Q --> E
    E --> VS
    VS --> R
    R --> CB
    CB --> BP
    BP --> LLM
    LLM --> ANS
    R --> CIT
```

## Flow

1. **embed_query** — Encode query with BGE-M3
2. **vector_search** — Top-K similarity in Weaviate
3. **build_context** — Assemble chunks (token limit, max chunks)
4. **build_prompt** — Structured prompt with system instruction + context + query
5. **LLM.generate** — Ollama completion
6. **Citations** — Derived from `(document_id, page_number)` of retrieved chunks

## Modules

- `src.retrieval.retriever` — `retrieve(query, top_k)`
- `src.rag.rag_pipeline` — `answer_query(query, llm, top_k)`
- `src.rag.prompt_template` — `build_prompt(context, query)`
