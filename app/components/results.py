"""Results display — metrics row, side-by-side comparison, and chunks."""

import html
from typing import Any

import streamlit as st


def _esc(s: str) -> str:
    return html.escape(str(s)) if s else ""


def render_metrics_row(bl: dict[str, Any], rag: dict[str, Any]) -> None:
    """Render the top-level metrics cards."""
    st.markdown(
        f"""
<div class="metrics-row">
  <div class="metric-card">
    <div class="metric-label">Fontes recuperadas</div>
    <div class="metric-value">{rag["sources"]}</div>
    <div class="metric-delta">&uarr; vs 0 no baseline</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Citacoes</div>
    <div class="metric-value">{len(rag.get("citations", []))}</div>
    <div class="metric-delta">documentos rastreaveis</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Latencia Baseline</div>
    <div class="metric-value">{bl["latency_ms"]} ms</div>
    <div class="metric-delta neutral">sem retrieval</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Latencia RAG</div>
    <div class="metric-value">{rag["latency_ms"]} ms</div>
    <div class="metric-delta neutral">embed + busca + geracao</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_comparison(bl: dict[str, Any], rag: dict[str, Any]) -> None:
    """Render side-by-side Baseline vs RAG comparison cards."""
    bl_ans = _esc(bl["answer"])
    rag_ans = _esc(rag["answer"])

    col_bl, col_vs, col_rag = st.columns([10, 1, 10])

    with col_bl:
        st.markdown(
            f"""
<div class="col-card baseline">
  <div class="col-header">
    <span class="col-title baseline">Baseline LLM</span>
    <span class="badge badge-red">Sem Retrieval</span>
  </div>
  <div class="response-text">{bl_ans}</div>
  <div class="meta-row">
    <div class="meta-item">
      <span class="meta-label">Modelo</span>
      <span class="meta-value">{_esc(bl["model"])}</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">Fontes</span>
      <span class="meta-value">0 documentos</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">Latencia</span>
      <span class="meta-value">{bl["latency_ms"]} ms</span>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col_vs:
        st.markdown(
            '<div class="vs-divider" style="height:100%;display:flex;align-items:center;">VS</div>',
            unsafe_allow_html=True,
        )

    with col_rag:
        citations_html = "".join(
            f'<span class="citation">{_esc(c)}</span>'
            for c in rag.get("citations", [])
        )
        st.markdown(
            f"""
<div class="col-card rag">
  <div class="col-header">
    <span class="col-title rag">RAG Pipeline</span>
    <span class="badge badge-green">Com Retrieval</span>
  </div>
  <div class="response-text">{rag_ans}</div>
  <div style="margin-top:0.8rem;">
    <div class="meta-label" style="margin-bottom:0.35rem;">Citacoes</div>
    {citations_html if citations_html else '<span style="color:#94a3b8;font-size:0.82rem;">&mdash;</span>'}
  </div>
  <div class="meta-row">
    <div class="meta-item">
      <span class="meta-label">Modelo</span>
      <span class="meta-value">{_esc(rag["model"])}</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">Fontes</span>
      <span class="meta-value">{rag["sources"]} chunks</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">Latencia</span>
      <span class="meta-value">{rag["latency_ms"]} ms</span>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )


def render_chunks(rag: dict[str, Any]) -> None:
    """Render retrieved chunks in an expandable section."""
    chunks = rag.get("chunks", [])
    if not chunks:
        return

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander(
        f"Contexto recuperado  --  {len(chunks)} chunks indexados", expanded=False
    ):
        for i, chunk in enumerate(chunks):
            score = chunk.get("score", 0)
            if score >= 0.94:
                score_color = "#16a34a"
            elif score >= 0.88:
                score_color = "#d97706"
            else:
                score_color = "#dc2626"

            doc_label = _esc(chunk.get("document_alias") or chunk.get("document_id", ""))
            page = chunk.get("page", "")
            section = _esc(chunk.get("section", ""))
            text = _esc(chunk.get("text", ""))
            st.markdown(
                f"""
<div class="chunk-card">
  <div class="chunk-header">
    <span class="chunk-tag">#{i + 1}</span>
    <span class="chunk-tag">{doc_label}</span>
    <span class="chunk-tag">p. {page}</span>
    <span class="chunk-tag">{section}</span>
    <span style="margin-left:auto;font-family:'IBM Plex Mono',monospace;
                 font-size:0.68rem;color:{score_color};font-weight:600;">
      {score:.3f}
    </span>
  </div>
  <div class="chunk-text">{text}</div>
</div>
""",
                unsafe_allow_html=True,
            )
