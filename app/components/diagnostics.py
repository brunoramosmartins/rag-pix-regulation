"""Retrieval diagnostics panel — score analysis, strategy info, document coverage."""

import html
from typing import Any

import streamlit as st


def _esc(s: str) -> str:
    return html.escape(str(s)) if s else ""


def _score_color(score: float) -> str:
    if score >= 0.8:
        return "#16a34a"
    if score >= 0.5:
        return "#d97706"
    return "#dc2626"


def render_diagnostics(rag: dict[str, Any]) -> None:
    """Render the retrieval diagnostics panel below the comparison."""
    chunks = rag.get("chunks", [])
    if not chunks:
        return

    scores = [c.get("score", 0) for c in chunks]
    max_score = max(scores) if scores else 0
    min_score = min(scores) if scores else 0
    mean_score = sum(scores) / len(scores) if scores else 0

    # Unique documents and pages
    unique_docs = {c.get("document_alias") or c.get("document_id") for c in chunks}
    unique_pages = {(c.get("document_id"), c.get("page")) for c in chunks}

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Diagnosticos de Retrieval", expanded=False):
        # Score statistics grid
        st.markdown(
            f"""
<div class="diag-section">
  <div class="diag-title">Estatisticas de Score</div>
  <div class="diag-grid">
    <div class="diag-item">
      <div class="diag-item-label">Score Maximo</div>
      <div class="diag-item-value" style="color:{_score_color(max_score)}">{max_score:.4f}</div>
    </div>
    <div class="diag-item">
      <div class="diag-item-label">Score Minimo</div>
      <div class="diag-item-value" style="color:{_score_color(min_score)}">{min_score:.4f}</div>
    </div>
    <div class="diag-item">
      <div class="diag-item-label">Score Medio</div>
      <div class="diag-item-value" style="color:{_score_color(mean_score)}">{mean_score:.4f}</div>
    </div>
    <div class="diag-item">
      <div class="diag-item-label">Total Chunks</div>
      <div class="diag-item-value">{len(chunks)}</div>
    </div>
    <div class="diag-item">
      <div class="diag-item-label">Documentos Unicos</div>
      <div class="diag-item-value">{len(unique_docs)}</div>
    </div>
    <div class="diag-item">
      <div class="diag-item-label">Paginas Unicas</div>
      <div class="diag-item-value">{len(unique_pages)}</div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # Per-chunk score bars
        st.markdown(
            '<div class="diag-section" style="margin-top:0.75rem;">'
            '<div class="diag-title">Distribuicao de Scores por Chunk</div>',
            unsafe_allow_html=True,
        )
        for i, chunk in enumerate(chunks):
            score = chunk.get("score", 0)
            bar_width = max(score * 100, 2)  # min 2% for visibility
            color = _score_color(score)
            doc_label = _esc(chunk.get("document_alias") or chunk.get("document_id", ""))
            page = chunk.get("page", "")
            st.markdown(
                f"""
<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.4rem;">
  <span style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#94a3b8;
               min-width:18px;text-align:right;">#{i+1}</span>
  <div style="flex:1;">
    <div class="score-bar">
      <div class="score-bar-fill" style="width:{bar_width}%;background:{color};"></div>
    </div>
  </div>
  <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
               color:{color};font-weight:600;min-width:50px;">{score:.4f}</span>
  <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
               color:#94a3b8;">{doc_label} p.{page}</span>
</div>
""",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Search strategy info
        try:
            from src.config import get_settings

            settings = get_settings()
            strategy = settings.retrieval.search_strategy
            alpha = settings.retrieval.hybrid.alpha
            fusion = settings.retrieval.hybrid.fusion_type
            reranking = settings.reranking.enabled
            rerank_model = settings.reranking.model

            st.markdown(
                f"""
<div class="diag-section" style="margin-top:0.75rem;">
  <div class="diag-title">Configuracao de Retrieval</div>
  <div class="diag-grid">
    <div class="diag-item">
      <div class="diag-item-label">Estrategia</div>
      <div class="diag-item-value">{_esc(strategy)}</div>
    </div>
    <div class="diag-item">
      <div class="diag-item-label">Alpha (Hybrid)</div>
      <div class="diag-item-value">{alpha}</div>
    </div>
    <div class="diag-item">
      <div class="diag-item-label">Fusion Type</div>
      <div class="diag-item-value">{_esc(fusion)}</div>
    </div>
    <div class="diag-item">
      <div class="diag-item-label">Reranking</div>
      <div class="diag-item-value">{"Ativo" if reranking else "Inativo"}</div>
    </div>
    <div class="diag-item">
      <div class="diag-item-label">Reranker Model</div>
      <div class="diag-item-value" style="font-size:0.75rem;">{_esc(rerank_model)}</div>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        except Exception:
            pass
