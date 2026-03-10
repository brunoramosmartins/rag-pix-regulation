# System Overview

High-level flow from user query to grounded response.

```mermaid
flowchart TB
    subgraph User
        Q[User Query]
    end

    subgraph RAG["RAG Pipeline"]
        E[Embed Query<br/>BGE-M3]
        V[Vector Search<br/>Weaviate]
        C[Context Builder]
        P[Prompt Builder]
        L[LLM<br/>Ollama]
    end

    subgraph Output
        R[Grounded Answer<br/>+ Citations]
    end

    Q --> E
    E --> V
    V --> C
    C --> P
    P --> L
    L --> R
```

## Components

| Step | Module | Description |
|------|--------|-------------|
| Embed | `retrieval.query_embedding` | Encode query with BGE-M3 |
| Search | `retrieval.vector_search` | Top-K similarity search in Weaviate |
| Context | `rag.context_builder` | Assemble chunks with token limit |
| Prompt | `rag.prompt_template` | Build structured prompt with context |
| LLM | `llm.baseline_llm` | Generate response via Ollama |
