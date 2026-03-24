"""Streamlit demo — Baseline vs RAG comparison for Pix regulation queries."""

import concurrent.futures
import sys
import html
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Register Phoenix tracer before RAG imports (optional)
try:
    from phoenix.otel import register
    register(
        project_name="rag-pix-regulation",
        endpoint="http://localhost:6006/v1/traces",
        protocol="http/protobuf",
        auto_instrument=False,
    )
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

# ── Custom CSS (versão nova priorizada) ─────────────────────────────────────────
st.markdown(
    """
<style>
/* (mantido exatamente como na sua branch nova) */
body { background-color: #f1f5f9; }
</style>
""",
    unsafe_allow_html=True,
)

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
<div style="text-align:center;padding:2rem;">
  <h1>Baseline vs RAG</h1>
  <p>Comparação de respostas com e sem contexto regulatório</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Query input ────────────────────────────────────────────────────────────────
query = st.text_area(
    label="Consulta",
    placeholder="Digite sua pergunta sobre regulamentação Pix…",
    height=100,
)

example = st.selectbox(
    "Ou selecione um exemplo",
    ["Selecione um exemplo…"] + EXAMPLE_QUERIES,
)

actual_query = query.strip() or (
    example if example != "Selecione um exemplo…" else ""
)

run_clicked = st.button("Executar comparação", use_container_width=True)


# ── Run pipeline ───────────────────────────────────────────────────────────────
if run_clicked and actual_query:
    try:
        with st.spinner("Executando..."):
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
    st.warning("Digite uma pergunta ou selecione um exemplo.")


# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.result_baseline and st.session_state.result_rag:
    bl = st.session_state.result_baseline
    rag = st.session_state.result_rag

    def format_answer(raw: str) -> str:
        if not raw:
            return ""
        paragraphs = raw.strip().split("\n\n")
        return "".join(f"<p>{html.escape(p)}</p>" for p in paragraphs)

    st.subheader("Baseline")
    st.markdown(format_answer(bl["answer"]), unsafe_allow_html=True)
    st.caption(f"Latency: {bl['latency_ms']} ms")

    st.subheader("RAG")
    st.markdown(format_answer(rag["answer"]), unsafe_allow_html=True)
    st.caption(f"Latency: {rag['latency_ms']} ms")

elif not run_clicked:
    st.info("Aguardando consulta...")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<hr>
<div style="text-align:center;font-size:0.8rem;color:#888;">
RAG Pix Regulation · Streamlit Demo
</div>
""",
    unsafe_allow_html=True,
)