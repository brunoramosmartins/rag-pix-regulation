"""Hero header and page configuration."""

import streamlit as st


def render_header() -> None:
    """Render the hero section with project branding and tech badges."""
    st.markdown(
        """
<div class="hero">
  <div class="hero-eyebrow">Regulamentacao Pix &middot; Retrieval-Augmented Generation</div>
  <h1 class="hero-title">
    <span class="baseline-word">Baseline LLM</span>
    <span class="vs-word"> vs </span>
    <span class="rag-word">RAG Pipeline</span>
  </h1>
  <p class="hero-subtitle">
    Compare respostas sem contexto (alucinacoes) versus respostas fundamentadas
    em documentos regulatorios oficiais do BCB com citacoes verificaveis.
  </p>
  <div class="hero-badges">
    <span class="hero-badge">BAAI/bge-m3</span>
    <span class="hero-badge">Weaviate</span>
    <span class="hero-badge">Llama 3.2 &middot; Ollama</span>
    <span class="hero-badge">Phoenix Tracing</span>
    <span class="hero-badge">Hybrid Search</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
