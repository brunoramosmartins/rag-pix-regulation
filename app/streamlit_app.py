"""Streamlit demo — Baseline vs RAG comparison for Pix regulation queries."""

import concurrent.futures
import html
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Register Phoenix tracer before RAG imports (optional; traces at http://localhost:6006)
try:
    from phoenix.otel import register
    register(project_name="rag-pix-regulation", auto_instrument=False)
except ImportError:
    pass

import streamlit as st

from src.demo import get_demo_health, run_baseline_query, run_rag_query

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG vs Baseline · Pix Regulation",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS — light professional theme ─────────────────────────────────────
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #f1f5f9;
    color: #0f172a;
  }
  .stApp { background-color: #f1f5f9; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }

  /* ── Hero ── */
  .hero {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    text-align: center;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  .hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #2563eb;
    margin-bottom: 0.6rem;
    font-weight: 500;
  }
  .hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #0f172a;
    margin: 0;
    line-height: 1.15;
  }
  .hero-title .baseline-word { color: #dc2626; }
  .hero-title .rag-word      { color: #16a34a; }
  .hero-title .vs-word       { color: #94a3b8; font-weight: 300; }
  .hero-subtitle {
    margin-top: 0.65rem;
    font-size: 0.92rem;
    color: #64748b;
    font-weight: 400;
  }
  .hero-badges {
    display: flex;
    gap: 0.5rem;
    justify-content: center;
    margin-top: 1rem;
    flex-wrap: wrap;
  }
  .hero-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    padding: 0.2rem 0.65rem;
    border-radius: 100px;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    color: #475569;
    letter-spacing: 0.08em;
  }

  /* ── Query box ── */
  .query-wrapper {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .query-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.63rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.4rem;
    font-weight: 500;
  }
  .stTextArea textarea {
    background: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
    resize: none !important;
  }
  .stTextArea textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
  }
  .stTextArea textarea::placeholder { color: #94a3b8 !important; }

  /* ── Selectbox ── */
  .stSelectbox [data-baseweb="select"] > div {
    background: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color: #475569 !important;
    font-size: 0.88rem !important;
  }

  /* ── Run button ── */
  .stButton > button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 2rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
    box-shadow: 0 1px 3px rgba(37,99,235,0.3) !important;
  }
  .stButton > button:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.35) !important;
    transform: translateY(-1px) !important;
  }
  .stButton > button:active { transform: translateY(0) !important; }

  /* ── Column cards ── */
  .col-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.5rem;
    height: 100%;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }
  .col-card.baseline { border-top: 4px solid #dc2626; }
  .col-card.rag      { border-top: 4px solid #16a34a; }

  /* ── Column header ── */
  .col-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.1rem;
    padding-bottom: 0.9rem;
    border-bottom: 1px solid #f1f5f9;
  }
  .col-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 600;
  }
  .col-title.baseline { color: #dc2626; }
  .col-title.rag      { color: #16a34a; }

  .badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    padding: 0.2rem 0.55rem;
    border-radius: 100px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .badge-red   { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
  .badge-green { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }

  /* ── Response text ── */
  .response-text {
    font-size: 0.91rem;
    line-height: 1.8;
    color: #1e293b;
    min-height: 120px;
    white-space: pre-wrap;
  }

  /* ── Meta row ── */
  .meta-row {
    display: flex;
    gap: 1.5rem;
    margin-top: 1.1rem;
    padding-top: 0.9rem;
    border-top: 1px solid #f1f5f9;
    flex-wrap: wrap;
  }
  .meta-item { display: flex; flex-direction: column; gap: 0.15rem; }
  .meta-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #94a3b8;
  }
  .meta-value { font-size: 0.82rem; color: #475569; font-weight: 500; }

  /* ── Chunk card ── */
  .chunk-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #2563eb;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.65rem;
  }
  .chunk-header {
    display: flex;
    gap: 0.4rem;
    align-items: center;
    margin-bottom: 0.45rem;
    flex-wrap: wrap;
  }
  .chunk-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: #dbeafe;
    color: #1d4ed8;
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    font-weight: 500;
  }
  .chunk-text {
    font-size: 0.83rem;
    color: #475569;
    line-height: 1.65;
  }

  /* ── Citations ── */
  .citation {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #15803d;
    border-radius: 4px;
    padding: 0.1rem 0.45rem;
    margin: 0.15rem 0.15rem 0.15rem 0;
    font-weight: 500;
  }

  /* ── VS divider ── */
  .vs-divider {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 0.5rem;
    color: #cbd5e1;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    font-weight: 500;
  }

  /* ── Metrics row ── */
  .metrics-row {
    display: flex;
    gap: 0.85rem;
    margin: 1.25rem 0 0.75rem;
    flex-wrap: wrap;
  }
  .metric-card {
    flex: 1;
    min-width: 140px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.35rem;
  }
  .metric-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
  }
  .metric-delta { font-size: 0.68rem; color: #16a34a; margin-top: 0.2rem; }
  .metric-delta.neutral { color: #64748b; }

  /* ── Expander ── */
  .streamlit-expanderHeader {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #475569 !important;
    font-size: 0.83rem !important;
  }
  .streamlit-expanderContent {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
  }

  /* ── Spinner ── */
  .stSpinner > div { border-top-color: #2563eb !important; }

  /* ── Alert boxes ── */
  .stWarning { background: #fffbeb !important; border-color: #fde68a !important; }
  .stInfo    { background: #eff6ff !important; border-color: #bfdbfe !important; }
  .stError   { background: #fef2f2 !important; border-color: #fecaca !important; }

  /* ── Footer ── */
  .footer {
    text-align: center;
    margin-top: 3.5rem;
    padding-top: 1.25rem;
    border-top: 1px solid #e2e8f0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    color: #94a3b8;
    text-transform: uppercase;
  }
</style>
""",
    unsafe_allow_html=True,
)

# ── Health check ───────────────────────────────────────────────────────────────
ready, msg = get_demo_health()
if not ready:
    st.warning(f"⚠️ **Dependências não encontradas:** {msg}")
    st.info(
        "Inicie o Weaviate (`docker compose up -d`), execute o pipeline "
        "(`python scripts/run_pipeline.py`) e suba o Ollama (`ollama serve`)."
    )

# ── Session state ──────────────────────────────────────────────────────────────
for key in ("result_baseline", "result_rag", "query_used"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Example queries ────────────────────────────────────────────────────────────
EXAMPLE_QUERIES = [
    "Como funciona o registro de chave Pix?",
    "Quais são as regras de devolução por fraude?",
    "Como funciona a portabilidade de chave Pix?",
    "Quantas chaves Pix posso ter por conta?",
    "O que é o DICT no contexto do Pix?",
    "Quais são os estados possíveis de uma notificação de infração no DICT?",
    "Qual é o prazo para iniciar uma solicitação de devolução após fraude?",
]

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
  <div class="hero-eyebrow">🏦 Regulamentação Pix · Retrieval-Augmented Generation</div>
  <h1 class="hero-title">
    <span class="baseline-word">Baseline LLM</span>
    <span class="vs-word"> vs </span>
    <span class="rag-word">RAG Pipeline</span>
  </h1>
  <p class="hero-subtitle">
    Compare respostas sem contexto (alucinações) versus respostas fundamentadas
    em documentos regulatórios oficiais do BCB com citações verificáveis.
  </p>
  <div class="hero-badges">
    <span class="hero-badge">BAAI/bge-m3</span>
    <span class="hero-badge">Weaviate</span>
    <span class="hero-badge">Llama 3.2 · Ollama</span>
    <span class="hero-badge">Phoenix Tracing</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Query input ────────────────────────────────────────────────────────────────
st.markdown('<div class="query-wrapper">', unsafe_allow_html=True)
st.markdown('<div class="query-label">Consulta</div>', unsafe_allow_html=True)

query = st.text_area(
    label="",
    placeholder="Digite sua pergunta sobre regulamentação Pix…",
    height=85,
    key="query_input",
    label_visibility="collapsed",
)

st.markdown(
    '<div class="query-label" style="margin-top:0.65rem;">Ou selecione um exemplo</div>',
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

run_clicked = st.button("🔍  Executar comparação", use_container_width=True)

# ── Run pipeline — parallel execution ──────────────────────────────────────────
if run_clicked and actual_query:
    try:
        with st.spinner("Executando Baseline e RAG em paralelo…"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_baseline = executor.submit(run_baseline_query, actual_query)
                future_rag = executor.submit(run_rag_query, actual_query)
                baseline = future_baseline.result()
                rag = future_rag.result()
        st.session_state.result_baseline = baseline
        st.session_state.result_rag = rag
        st.session_state.query_used = actual_query
    except Exception as e:
        st.error(f"Erro ao executar: {e}")
        st.exception(e)
elif run_clicked:
    st.warning("Digite uma pergunta ou selecione um exemplo antes de executar.")

# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.result_baseline and st.session_state.result_rag:
    bl = st.session_state.result_baseline
    rag = st.session_state.result_rag

    def _esc(s: str) -> str:
        return html.escape(str(s)) if s else ""

    bl_ans = _esc(bl["answer"])
    rag_ans = _esc(rag["answer"])

    # ── Metrics row ────────────────────────────────────────────────────────────
    latency_diff = rag["latency_ms"] - bl["latency_ms"]
    latency_delta_color = "#64748b"  # neutral — RAG always slower due to retrieval
    st.markdown(
        f"""
<div class="metrics-row">
  <div class="metric-card">
    <div class="metric-label">Fontes recuperadas</div>
    <div class="metric-value">{rag["sources"]}</div>
    <div class="metric-delta">↑ vs 0 no baseline</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Citações</div>
    <div class="metric-value">{len(rag.get("citations", []))}</div>
    <div class="metric-delta">documentos rastreáveis</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Latência Baseline</div>
    <div class="metric-value">{bl["latency_ms"]} ms</div>
    <div class="metric-delta neutral">sem retrieval</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Latência RAG</div>
    <div class="metric-value">{rag["latency_ms"]} ms</div>
    <div class="metric-delta neutral">embed + busca + geração</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Side-by-side ──────────────────────────────────────────────────────────
    col_bl, col_vs, col_rag = st.columns([10, 1, 10])

    with col_bl:
        st.markdown(
            f"""
<div class="col-card baseline">
  <div class="col-header">
    <span class="col-title baseline">⚠ Baseline LLM</span>
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
      <span class="meta-label">Latência</span>
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
    <span class="col-title rag">✓ RAG Pipeline</span>
    <span class="badge badge-green">Com Retrieval</span>
  </div>
  <div class="response-text">{rag_ans}</div>
  <div style="margin-top:0.8rem;">
    <div class="meta-label" style="margin-bottom:0.35rem;">Citações</div>
    {citations_html if citations_html else '<span style="color:#94a3b8;font-size:0.82rem;">—</span>'}
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
      <span class="meta-label">Latência</span>
      <span class="meta-value">{rag["latency_ms"]} ms</span>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ── Retrieved chunks ───────────────────────────────────────────────────────
    chunks = rag.get("chunks", [])
    if chunks:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander(
            f"📄  Contexto recuperado — {len(chunks)} chunks indexados", expanded=False
        ):
            for i, chunk in enumerate(chunks):
                score = chunk.get("score", 0)
                if score >= 0.94:
                    score_color = "#16a34a"
                elif score >= 0.88:
                    score_color = "#d97706"
                else:
                    score_color = "#dc2626"

                doc_label = _esc(chunk.get("document_alias") or chunk.get("document_id", "—"))
                page = chunk.get("page", "—")
                section = _esc(chunk.get("section", "—"))
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

# ── Empty state ────────────────────────────────────────────────────────────────
elif not run_clicked:
    st.markdown(
        """
<div style="text-align:center;padding:4rem 0;color:#cbd5e1;">
  <div style="font-size:3rem;margin-bottom:0.75rem;">🏦</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
              letter-spacing:0.22em;text-transform:uppercase;color:#94a3b8;">
    Aguardando consulta
  </div>
  <div style="font-size:0.85rem;margin-top:0.5rem;color:#cbd5e1;">
    Selecione um exemplo ou escreva uma pergunta e clique em <strong>Executar comparação</strong>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="footer">
  RAG Pix Regulation &nbsp;·&nbsp; Weaviate &nbsp;·&nbsp; Ollama &nbsp;·&nbsp; Streamlit
  &nbsp;|&nbsp; Base de conhecimento: Regulamentação Pix BCB · MED 2.0
</div>
""",
    unsafe_allow_html=True,
)
