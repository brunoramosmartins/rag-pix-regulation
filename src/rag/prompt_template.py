"""Structured prompt template for RAG with clear delimiters."""

SYSTEM_INSTRUCTION = """You are an assistant specialized in Brazilian Pix regulation.

Use only the provided regulatory context.

If the answer is not present in the context, say that the information is not available."""


def build_prompt(context: str, query: str) -> str:
    """
    Build a structured RAG prompt with explicit sections.
    """
    return f"""---

System Instruction

{SYSTEM_INSTRUCTION}

---

Regulatory Context

{context}

---

User Question

{query}

---

Answer

"""
