"""Streamlit demo — Baseline vs RAG comparison for Pix regulation queries."""

import concurrent.futures
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

from app.styles import CSS
from app.components.header import render_header
from app.components.query_input import render_query_input
from app.components.results import render_metrics_row, render_comparison, render_chunks
from app.components.diagnostics import render_diagnostics
from app.components.evaluation import render_evaluation_mode
from src.demo import get_demo_health, run_baseline_query, run_rag_query

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG vs Baseline · Pix Regulation",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject CSS ─────────────────────────────────────────────────────────────────
st.markdown(CSS, unsafe_allow_html=True)

# ── Health check ───────────────────────────────────────────────────────────────
ready, msg = get_demo_health()
if not ready:
    st.warning(f"**Dependencias nao encontradas:** {msg}")
    st.info(
        "Inicie o Weaviate (`docker compose up -d`), execute o pipeline "
        "(`python scripts/run_pipeline.py`) e suba o Ollama (`ollama serve`)."
    )

# ── Session state ──────────────────────────────────────────────────────────────
for key in ("result_baseline", "result_rag", "query_used"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Sidebar — mode toggle ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="diag-title" style="margin-bottom:1rem;">Modo</div>',
        unsafe_allow_html=True,
    )
    app_mode = st.radio(
        "Selecione o modo",
        options=["Comparacao", "Avaliacao"],
        index=0,
        key="app_mode",
        label_visibility="collapsed",
    )

# ── Header ─────────────────────────────────────────────────────────────────────
render_header()

# ── Main content by mode ───────────────────────────────────────────────────────
if app_mode == "Avaliacao":
    render_evaluation_mode()
else:
    # ── Query input ────────────────────────────────────────────────────────────
    actual_query, run_clicked = render_query_input()

    # ── Run pipeline — parallel execution ──────────────────────────────────────
    if run_clicked and actual_query:
        try:
            with st.spinner("Executando Baseline e RAG em paralelo..."):
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

    # ── Results ────────────────────────────────────────────────────────────────
    if st.session_state.result_baseline and st.session_state.result_rag:
        bl = st.session_state.result_baseline
        rag = st.session_state.result_rag

        render_metrics_row(bl, rag)
        st.markdown("<br>", unsafe_allow_html=True)
        render_comparison(bl, rag)
        render_diagnostics(rag)
        render_chunks(rag)

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
    Selecione um exemplo ou escreva uma pergunta e clique em <strong>Executar comparacao</strong>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="footer">
  RAG Pix Regulation &nbsp;&middot;&nbsp; Weaviate &nbsp;&middot;&nbsp; Ollama &nbsp;&middot;&nbsp; Streamlit
  &nbsp;|&nbsp; Base de conhecimento: Regulamentacao Pix BCB &middot; MED 2.0
</div>
""",
    unsafe_allow_html=True,
)
