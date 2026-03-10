"""Demo module — service layer for interactive RAG demonstration."""

from .demo_service import get_demo_health, run_baseline_query, run_rag_query

__all__ = ["get_demo_health", "run_baseline_query", "run_rag_query"]
