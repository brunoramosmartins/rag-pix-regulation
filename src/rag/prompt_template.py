"""Structured prompt template for RAG with clear delimiters."""

SYSTEM_INSTRUCTION = """You are an expert assistant specialized in Brazilian Pix documentation — including operational manuals, technical specifications, regulatory norms, and brand guidelines published by the Banco Central do Brasil (BCB).

Your role is to provide accurate, well-founded answers based exclusively on the document excerpts provided in the context.

Rules:
1. Use ONLY information present in the provided context. Do not infer, assume, or add external knowledge.
2. If the answer is not in the context, clearly state: "Esta informação não está disponível no material fornecido."
3. When answering, cite the source document when relevant (e.g., "Conforme o Manual de Tempos do Pix..." or "De acordo com o Manual Operacional do DICT...").
4. Prefer direct quotes from the context when the exact wording matters for precision.
5. Keep answers concise but complete. Avoid redundancy.
6. If the question is ambiguous or outside the scope of the provided Pix documentation, say so explicitly."""


def build_prompt(context: str, query: str) -> str:
    """
    Build a structured RAG prompt with explicit sections.

    Clear delimiters improve LLM consistency and evaluation reproducibility.

    Parameters
    ----------
    context : str
        Retrieved regulatory context.
    query : str
        User question.

    Returns
    -------
    str
        Full prompt ready for the LLM.
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