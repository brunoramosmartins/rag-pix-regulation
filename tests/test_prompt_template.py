"""Unit tests for RAG prompt template."""

import importlib.util
from pathlib import Path

# Load prompt_template directly to avoid pulling in rag_pipeline (sentence_transformers)
_spec = importlib.util.spec_from_file_location(
    "prompt_template",
    Path(__file__).resolve().parent.parent / "src" / "rag" / "prompt_template.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_prompt = _mod.build_prompt


def test_build_prompt_includes_sections() -> None:
    """Prompt has clear structural delimiters."""
    prompt = build_prompt(context="Regulatory text here.", query="What is Pix?")
    assert "System Instruction" in prompt
    assert "Regulatory Context" in prompt
    assert "User Question" in prompt
    assert "Answer" in prompt
    assert "Regulatory text here." in prompt
    assert "What is Pix?" in prompt


def test_build_prompt_includes_system_instruction() -> None:
    """System instruction is embedded."""
    prompt = build_prompt(context="x", query="y")
    assert "Brazilian Pix regulation" in prompt
    assert "only the provided regulatory context" in prompt
    assert "not present" in prompt or "not available" in prompt


def test_build_prompt_deterministic() -> None:
    """Same inputs produce same prompt."""
    p1 = build_prompt("ctx", "q")
    p2 = build_prompt("ctx", "q")
    assert p1 == p2
