"""Streamlit demo — Baseline vs RAG comparison for Pix regulation queries."""

import html
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.demo import get_demo_health, run_baseline_query, run_rag_query

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG vs Baseline · Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  /* ── Reset & base ── */
  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #09090b;
    color: #e4e4e7;
  }

  .stApp { background-color: #09090b; }

  /* ── Hide default streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }

  /* ── Hero header ── */
  .hero {
    text-align: center;
    padding: 3rem 0 2rem;
    border-bottom: 1px solid #27272a;
    margin-bottom: 2.5rem;
  }
  .hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #71717a;
    margin-bottom: 0.75rem;
  }
  .hero-title {
    font-size: 2.6rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    color: #fafafa;
    margin: 0;
    line-height: 1.1;
  }
  .hero-title span.accent { color: #f59e0b; }
  .hero-subtitle {
    margin-top: 0.75rem;
    font-size: 0.95rem;
    color: #71717a;
    font-weight: 300;
  }

  /* ── Query box ── */
  .query-wrapper {
    background: #18181b;
    border: 1px solid #3f3f46;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 2rem;
  }
  .query-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #52525b;
    margin-bottom: 0.5rem;
  }
  .stTextArea textarea {
    background: #09090b !important;
    border: 1px solid #3f3f46 !important;
    border-radius: 8px !important;
    color: #e4e4e7 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
    resize: none !important;
  }
  .stTextArea textarea:focus {
    border-color: #f59e0b !important;
    box-shadow: 0 0 0 2px rgba(245,158,11,0.15) !important;
  }

  /* ── Run button ── */
  .stButton > button {
    background: #f59e0b !important;
    color: #09090b !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 2.2rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.15s !important;
    width: 100% !important;
  }
  .stButton > button:hover {
    background: #d97706 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(245,158,11,0.35) !important;
  }
  .stButton > button:active { transform: translateY(0) !important; }

  /* ── Column cards ── */
  .col-card {
    background: #18181b;
    border-radius: 12px;
    padding: 1.5rem;
    height: 100%;
    border: 1px solid #27272a;
  }
  .col-card.baseline { border-top: 3px solid #ef4444; }
  .col-card.rag      { border-top: 3px solid #22c55e; }

  /* ── Column header ── */
  .col-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.25rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #27272a;
  }
  .col-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 500;
  }
  .col-title.baseline { color: #ef4444; }
  .col-title.rag      { color: #22c55e; }

  .badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    padding: 0.2rem 0.6rem;
    border-radius: 100px;
    font-weight: 500;
    letter-spacing: 0.1em;
  }
  .badge-red   { background: rgba(239,68,68,0.12);  color: #f87171; border: 1px solid rgba(239,68,68,0.25); }
  .badge-green { background: rgba(34,197,94,0.12);  color: #4ade80; border: 1px solid rgba(34,197,94,0.25); }

  /* ── Response text ── */
  .response-text {
    font-size: 0.92rem;
    line-height: 1.75;
    color: #d4d4d8;
    min-height: 120px;
  }

  /* ── Meta row ── */
  .meta-row {
    display: flex;
    gap: 1.25rem;
    margin-top: 1.25rem;
    padding-top: 1rem;
    border-top: 1px solid #27272a;
  }
  .meta-item { display: flex; flex-direction: column; gap: 0.2rem; }
  .meta-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #52525b;
  }
  .meta-value { font-size: 0.82rem; color: #a1a1aa; }

  /* ── Chunk card ── */
  .chunk-card {
    background: #09090b;
    border: 1px solid #27272a;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
  }
  .chunk-header {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    margin-bottom: 0.5rem;
    flex-wrap: wrap;
  }
  .chunk-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    background: #1e3a5f;
    color: #93c5fd;
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
  }
  .chunk-text {
    font-size: 0.82rem;
    color: #a1a1aa;
    line-height: 1.6;
  }

  /* ── Citations ── */
  .citation {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    color: #4ade80;
    border-radius: 4px;
    padding: 0.1rem 0.45rem;
    margin: 0.15rem 0.15rem 0.15rem 0;
  }

  /* ── Divider ── */
  .vs-divider {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 0.5rem;
    color: #3f3f46;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
  }

  /* ── Metric pills ── */
  .metrics-row {
    display: flex;
    gap: 1rem;
    margin: 2rem 0 0.5rem;
    flex-wrap: wrap;
  }
  .metric-card {
    flex: 1;
    min-width: 160px;
    background: #18181b;
    border: 1px solid #27272a;
    border-radius: 10px;
    padding: 1rem 1.25rem;
  }
  .metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #52525b;
    margin-bottom: 0.4rem;
  }
  .metric-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #fafafa;
  }
  .metric-delta {
    font-size: 0.7rem;
    color: #22c55e;
    margin-top: 0.15rem;
  }

  /* ── Expander ── */
  .streamlit-expanderHeader {
    background: #18181b !important;
    border: 1px solid #27272a !important;
    border-radius: 8px !important;
    color: #a1a1aa !important;
    font-size: 0.82rem !important;
  }
  .streamlit-expanderContent {
    background: #18181b !important;
    border: 1px solid #27272a !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
  }

  /* ── Spinner override ── */
  .stSpinner > div { border-top-color: #f59e0b !important; }

  /* ── Footer ── */
  .footer {
    text-align: center;
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid #27272a;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: #3f3f46;
    text-transform: uppercase;
  }
</style>
""",
    unsafe_allow_html=True,
)

# ── Health check ───────────────────────────────────────────────────────────────
ready, msg = get_demo_health()
if not ready:
    st.warning(f"⚠️ **Dependencies not ready:** {msg}")
    st.info(
        "Ensure Weaviate is running (`docker compose up -d`), the pipeline is indexed "
        "(`python scripts/run_pipeline.py`), and Ollama is running (`ollama serve`)."
    )

# ── Session state ──────────────────────────────────────────────────────────────
if "result_baseline" not in st.session_state:
    st.session_state.result_baseline = None
if "result_rag" not in st.session_state:
    st.session_state.result_rag = None
if "query_used" not in st.session_state:
    st.session_state.query_used = ""

# ── Example queries ───────────────────────────────────────────────────────────
EXAMPLE_QUERIES = [
    "Como funciona o registro de chave Pix?",
    "Quais são as regras de devolução por fraude?",
    "Como funciona a portabilidade de chave Pix?",
    "Quantas chaves Pix posso ter por conta?",
    "O que é o DICT no contexto do Pix?",
]

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
  <div class="hero-eyebrow">Retrieval-Augmented Generation · Demo</div>
  <h1 class="hero-title">Baseline LLM <span class="accent">vs</span> RAG Pipeline</h1>
  <p class="hero-subtitle">
    Compare grounded, citation-backed answers against unaugmented generation &mdash;
    Pix regulation knowledge base · Weaviate + Ollama
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Query input ─────────────────────────────────────────────────────────────────
st.markdown('<div class="query-wrapper">', unsafe_allow_html=True)
st.markdown('<div class="query-label">Consulta</div>', unsafe_allow_html=True)

query = st.text_area(
    label="",
    placeholder="Digite sua pergunta sobre regulamentação Pix…",
    height=90,
    key="query_input",
    label_visibility="collapsed",
)

st.markdown(
    '<div class="query-label" style="margin-top:0.75rem;">Exemplos</div>',
    unsafe_allow_html=True,
)
example = st.selectbox(
    label="",
    options=["Selecione um exemplo…"] + EXAMPLE_QUERIES,
    key="example_select",
    label_visibility="collapsed",
)
actual_query = query.strip() or (
    example if example and example != "Selecione um exemplo…" else ""
)

st.markdown("</div>", unsafe_allow_html=True)

run_clicked = st.button("⚡  Executar comparação", use_container_width=True)

# ── Run pipeline ───────────────────────────────────────────────────────────────
if run_clicked and actual_query:
    try:
        with st.spinner("Executando pipeline…"):
            baseline = run_baseline_query(actual_query)
            rag = run_rag_query(actual_query)
        st.session_state.result_baseline = baseline
        st.session_state.result_rag = rag
        st.session_state.query_used = actual_query
    except Exception as e:
        st.error(f"Erro ao executar: {e}")
        st.exception(e)
elif run_clicked:
    st.warning("Digite uma pergunta ou selecione um exemplo antes de executar.")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.result_baseline and st.session_state.result_rag:
    bl = st.session_state.result_baseline
    rag = st.session_state.result_rag

    # ── Escape user content for HTML ─────────────────────────────────────────────
    def _esc(s: str) -> str:
        return html.escape(str(s)) if s else ""

    bl_ans, rag_ans = _esc(bl["answer"]), _esc(rag["answer"])
    bl_mod, rag_mod = _esc(bl["model"]), _esc(rag["model"])

    # ── Top metrics row ────────────────────────────────────────────────────────
    st.markdown(
        f"""
    <div class="metrics-row">
      <div class="metric-card">
        <div class="metric-label">Fontes recuperadas</div>
        <div class="metric-value">{rag["sources"]}</div>
        <div class="metric-delta">↑ vs 0 (baseline)</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Latência baseline</div>
        <div class="metric-value">{bl["latency_ms"]} ms</div>
        <div class="metric-delta" style="color:#a1a1aa;">sem retrieval</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Latência RAG</div>
        <div class="metric-value">{rag["latency_ms"]} ms</div>
        <div class="metric-delta">embed + search + gen</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Citações</div>
        <div class="metric-value">{len(rag.get("citations", []))}</div>
        <div class="metric-delta">documentos rastreáveis</div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Side-by-side columns ───────────────────────────────────────────────────
    col_bl, col_vs, col_rag = st.columns([10, 1, 10])

    # ── Baseline ──────────────────────────────────────────────────────────────
    with col_bl:
        st.markdown(
            f"""
        <div class="col-card baseline">
          <div class="col-header">
            <span class="col-title baseline">⬡ Baseline LLM</span>
            <span class="badge badge-red">SEM RETRIEVAL</span>
          </div>
          <div class="response-text">{bl_ans}</div>
          <div class="meta-row">
            <div class="meta-item">
              <span class="meta-label">Modelo</span>
              <span class="meta-value">{bl_mod}</span>
            </div>
            <div class="meta-item">
              <span class="meta-label">Fontes</span>
              <span class="meta-value">{bl["sources"]} documentos</span>
            </div>
            <div class="meta-item">
              <span class="meta-label">Latência</span>
              <span class="meta-value">{bl["latency_ms"]} ms</span>
            </div>
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── VS divider ──────────────────────────────────────────────────────────────
    with col_vs:
        st.markdown(
            '<div class="vs-divider" style="height:100%;display:flex;align-items:center;">VS</div>',
            unsafe_allow_html=True,
        )

    # ── RAG ────────────────────────────────────────────────────────────────────
    with col_rag:
        citations_html = "".join(
            f'<span class="citation">{_esc(c)}</span>' for c in rag.get("citations", [])
        )
        st.markdown(
            f"""
        <div class="col-card rag">
          <div class="col-header">
            <span class="col-title rag">◈ RAG Pipeline</span>
            <span class="badge badge-green">COM RETRIEVAL</span>
          </div>
          <div class="response-text">{rag_ans}</div>
          <div style="margin-top:0.85rem;">
            <div class="meta-label" style="margin-bottom:0.4rem;">Citações</div>
            {citations_html}
          </div>
          <div class="meta-row">
            <div class="meta-item">
              <span class="meta-label">Modelo</span>
              <span class="meta-value">{rag_mod}</span>
            </div>
            <div class="meta-item">
              <span class="meta-label">Fontes</span>
              <span class="meta-value">{rag["sources"]} chunks</span>
            </div>
            <div class="meta-item">
              <span class="meta-label">Latência</span>
              <span class="meta-value">{rag["latency_ms"]} ms</span>
            </div>
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── Retrieved context ──────────────────────────────────────────────────────
    chunks = rag.get("chunks", [])
    if chunks:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander(
            f"🔍  Contexto recuperado — {len(chunks)} chunks", expanded=False
        ):
            for i, chunk in enumerate(chunks):
                score = chunk.get("score", 0)
                score_color = (
                    "#22c55e"
                    if score >= 0.94
                    else "#f59e0b"
                    if score >= 0.88
                    else "#ef4444"
                )
                doc_id = _esc(chunk.get("document_id", "—"))
                page = chunk.get("page", "—")
                section = _esc(chunk.get("section", "—"))
                text = _esc(chunk.get("text", ""))
                st.markdown(
                    f"""
                <div class="chunk-card">
                  <div class="chunk-header">
                    <span class="chunk-tag">#{i + 1}</span>
                    <span class="chunk-tag">{doc_id}</span>
                    <span class="chunk-tag">p. {page}</span>
                    <span class="chunk-tag">{section}</span>
                    <span style="margin-left:auto;font-family:'IBM Plex Mono',monospace;
                                 font-size:0.7rem;color:{score_color};">
                      score {score:.3f}
                    </span>
                  </div>
                  <div class="chunk-text">{text}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

# ── Empty state ────────────────────────────────────────────────────────────────
elif not run_clicked:
    st.markdown(
        """
    <div style="text-align:center;padding:4rem 0;color:#3f3f46;">
      <div style="font-size:2.5rem;margin-bottom:1rem;">⚡</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;
                  letter-spacing:0.2em;text-transform:uppercase;">
        Aguardando consulta
      </div>
      <div style="font-size:0.85rem;margin-top:0.5rem;color:#27272a;">
        Digite uma pergunta acima ou selecione um exemplo e clique em Executar
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="footer">
  RAG Demo · Weaviate · Ollama · Streamlit &nbsp;|&nbsp;
  Knowledge base: Regulamentação Pix BCB
</div>
""",
    unsafe_allow_html=True,
)
