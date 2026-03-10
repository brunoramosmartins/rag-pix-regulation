#!/usr/bin/env python
"""
Compare RAG vs baseline LLM answers for the same query.

Run from project root (Ollama required):

    python scripts/compare_rag_vs_llm.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm import BaselineLLM
from src.rag.rag_pipeline import answer_query
from src.vectorstore.weaviate_client import is_weaviate_ready

QUERY = "Como funciona o cadastro de chave Pix?"


def main() -> None:
    """Compare RAG and baseline LLM outputs."""
    if not is_weaviate_ready():
        print("ERROR: Weaviate is not running. Run: python scripts/run_indexing.py")
        sys.exit(1)

    llm = BaselineLLM()

    print("\n" + "=" * 60)
    print("Query")
    print("=" * 60)
    print(QUERY)

    # RAG answer
    rag_response = answer_query(QUERY, llm=llm)
    print("\n" + "=" * 60)
    print("RAG Answer (with retrieved context)")
    print("=" * 60)
    print(rag_response.answer)
    print("\nCitations:", rag_response.citations)

    # Baseline LLM (no retrieval - answers from model knowledge)
    baseline_prompt = f"Pergunta: {QUERY}\n\nResposta:"
    baseline_answer = llm.generate(baseline_prompt)
    print("\n" + "=" * 60)
    print("Baseline LLM Answer (no retrieval)")
    print("=" * 60)
    print(baseline_answer)


if __name__ == "__main__":
    main()
