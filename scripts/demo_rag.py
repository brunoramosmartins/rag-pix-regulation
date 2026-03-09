#!/usr/bin/env python
"""
Demo: RAG pipeline with prompt logging for debugging.

Run from project root (after run_indexing.py and with Ollama running):

    python scripts/demo_rag.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm import BaselineLLM
from src.rag.rag_pipeline import answer_query
from src.vectorstore.weaviate_client import is_weaviate_ready

DEMO_QUERY = "Como funciona o cadastro de chave Pix?"


def main() -> None:
    """Run RAG demo with full prompt logging."""
    if not is_weaviate_ready():
        print("ERROR: Weaviate is not running. Run: python scripts/run_indexing.py")
        sys.exit(1)

    llm = BaselineLLM()
    response = answer_query(DEMO_QUERY, llm=llm, top_k=5, max_chunks=5)

    print("\n" + "=" * 60)
    print("Query")
    print("=" * 60)
    print(response.query)

    print("\n" + "=" * 60)
    print("Retrieved chunks")
    print("=" * 60)
    for r in response.retrieved_chunks:
        print(f"  {r.document_id} p.{r.page_number} (score: {r.similarity_score})")

    print("\n" + "=" * 60)
    print("Prompt preview (first 800 chars)")
    print("=" * 60)
    from src.rag.prompt_template import build_prompt

    prompt = build_prompt(response.context, response.query)
    print(prompt[:800] + "..." if len(prompt) > 800 else prompt)

    print("\n" + "=" * 60)
    print("Answer")
    print("=" * 60)
    print(response.answer)

    print("\n" + "=" * 60)
    print("Citations")
    print("=" * 60)
    for c in response.citations:
        print(f"  {c}")


if __name__ == "__main__":
    main()
