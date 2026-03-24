"""Evaluation mode — run dataset queries and display quality metrics."""

import html
import json
from pathlib import Path
from typing import Any

import streamlit as st

_DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation" / "rag_evaluation_dataset.json"

DIFFICULTY_LABELS = {
    "single_chunk": "Single Chunk",
    "multi_chunk": "Multi Chunk",
    "cross_section": "Cross Section",
    "negative": "Negative",
}


def _esc(s: str) -> str:
    return html.escape(str(s)) if s else ""


def _load_dataset() -> list[dict[str, Any]]:
    """Load evaluation dataset queries."""
    if not _DATASET_PATH.exists():
        return []
    with open(_DATASET_PATH) as f:
        data = json.load(f)
    return data.get("queries", [])


def render_evaluation_mode() -> None:
    """Render evaluation mode panel in the sidebar and results area."""
    queries = _load_dataset()
    if not queries:
        st.warning("Dataset de avaliacao nao encontrado.")
        return

    st.markdown(
        '<div class="diag-title" style="margin-bottom:0.5rem;">Modo de Avaliacao</div>',
        unsafe_allow_html=True,
    )

    # Difficulty filter
    difficulties = sorted({q["difficulty"] for q in queries})
    selected_diff = st.multiselect(
        "Filtrar por dificuldade",
        options=difficulties,
        default=difficulties,
        format_func=lambda d: DIFFICULTY_LABELS.get(d, d),
        key="eval_difficulty_filter",
    )

    filtered = [q for q in queries if q["difficulty"] in selected_diff]
    st.markdown(
        f'<div style="font-size:0.78rem;color:#64748b;margin-bottom:0.5rem;">'
        f'{len(filtered)} queries selecionadas de {len(queries)} total</div>',
        unsafe_allow_html=True,
    )

    # Select a query to evaluate
    query_options = {
        f'{q["query_id"]} — {q["query"][:60]}...': q for q in filtered
    }
    selected_label = st.selectbox(
        "Selecionar query",
        options=["Selecione uma query..."] + list(query_options.keys()),
        key="eval_query_select",
    )

    if selected_label == "Selecione uma query..." or selected_label not in query_options:
        _render_dataset_overview(filtered)
        return

    query_data = query_options[selected_label]

    run_eval = st.button("Executar avaliacao", key="eval_run_btn", use_container_width=True)

    # Show expected data
    _render_expected_data(query_data)

    if run_eval:
        _run_evaluation(query_data)


def _render_dataset_overview(queries: list[dict[str, Any]]) -> None:
    """Show overview of the evaluation dataset."""
    difficulty_counts = {}
    for q in queries:
        d = q["difficulty"]
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1

    grid_items = "".join(
        f"""
    <div class="diag-item">
      <div class="diag-item-label">{DIFFICULTY_LABELS.get(d, d)}</div>
      <div class="diag-item-value">{count}</div>
    </div>"""
        for d, count in sorted(difficulty_counts.items())
    )
    st.markdown(
        f"""
<div class="diag-section">
  <div class="diag-title">Dataset Overview</div>
  <div class="diag-grid">{grid_items}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_expected_data(query_data: dict[str, Any]) -> None:
    """Display expected answer and key concepts for a query."""
    difficulty = query_data.get("difficulty", "")
    expected = _esc(query_data.get("expected_answer_summary", ""))
    concepts = query_data.get("key_concepts", [])
    pages = query_data.get("expected_pages", [])

    concepts_html = " ".join(
        f'<span class="chunk-tag">{_esc(c)}</span>' for c in concepts
    )
    pages_html = " ".join(
        f'<span class="chunk-tag">p. {p}</span>' for p in pages
    )

    st.markdown(
        f"""
<div class="eval-card">
  <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.6rem;">
    <span class="eval-badge {difficulty}">{DIFFICULTY_LABELS.get(difficulty, difficulty)}</span>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#94a3b8;">
      {_esc(query_data.get("query_id", ""))}
    </span>
  </div>
  <div class="eval-query">{_esc(query_data.get("query", ""))}</div>
  <div style="margin-top:0.6rem;">
    <div class="meta-label" style="margin-bottom:0.3rem;">Resposta Esperada</div>
    <div style="font-size:0.82rem;color:#475569;line-height:1.6;">{expected}</div>
  </div>
  <div style="margin-top:0.5rem;">
    <div class="meta-label" style="margin-bottom:0.3rem;">Conceitos-Chave</div>
    <div>{concepts_html if concepts_html else '<span style="color:#94a3b8;">—</span>'}</div>
  </div>
  <div style="margin-top:0.5rem;">
    <div class="meta-label" style="margin-bottom:0.3rem;">Paginas Esperadas</div>
    <div>{pages_html if pages_html else '<span style="color:#94a3b8;">—</span>'}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _run_evaluation(query_data: dict[str, Any]) -> None:
    """Execute RAG pipeline and display evaluation metrics for the selected query."""
    query = query_data["query"]

    with st.spinner("Executando pipeline RAG e calculando metricas..."):
        try:
            from src.demo import run_rag_query
            from src.evaluation.answer_quality import (
                compute_concept_coverage,
            )
            from src.evaluation.rag_evaluation import (
                compute_citation_coverage,
                detect_hallucination,
            )

            rag_result = run_rag_query(query, top_k=10)
            answer = rag_result["answer"]
            chunks = rag_result.get("chunks", [])
            citations = rag_result.get("citations", [])

            # Compute metrics
            expected_pages = query_data.get("expected_pages", [])
            key_concepts = query_data.get("key_concepts", [])

            # Citation coverage
            retrieved_pairs = [
                (c.get("document_id", ""), c.get("page", 0)) for c in chunks
            ]
            citation_cov = compute_citation_coverage(citations, retrieved_pairs)

            # Concept coverage
            concept_cov = compute_concept_coverage(answer, key_concepts) if key_concepts else 0.0

            # Page recall
            retrieved_pages = {c.get("page") for c in chunks}
            page_recall = (
                len(set(expected_pages) & retrieved_pages) / len(expected_pages)
                if expected_pages
                else 0.0
            )

            # Hallucination check
            context = " ".join(c.get("text", "") for c in chunks)
            hallucination = detect_hallucination(answer, context)

            # Display results
            hall_color = "#dc2626" if hallucination else "#16a34a"
            hall_text = "Detectada" if hallucination else "Nao detectada"

            st.markdown(
                f"""
<div class="diag-section" style="margin-top:0.75rem;">
  <div class="diag-title">Resultados da Avaliacao</div>
  <div class="diag-grid">
    <div class="diag-item">
      <div class="diag-item-label">Cobertura de Citacoes</div>
      <div class="diag-item-value">{citation_cov:.0%}</div>
    </div>
    <div class="diag-item">
      <div class="diag-item-label">Cobertura de Conceitos</div>
      <div class="diag-item-value">{concept_cov:.0%}</div>
    </div>
    <div class="diag-item">
      <div class="diag-item-label">Recall de Paginas</div>
      <div class="diag-item-value">{page_recall:.0%}</div>
    </div>
    <div class="diag-item">
      <div class="diag-item-label">Alucinacao</div>
      <div class="diag-item-value" style="color:{hall_color}">{hall_text}</div>
    </div>
    <div class="diag-item">
      <div class="diag-item-label">Chunks Recuperados</div>
      <div class="diag-item-value">{len(chunks)}</div>
    </div>
    <div class="diag-item">
      <div class="diag-item-label">Latencia</div>
      <div class="diag-item-value">{rag_result["latency_ms"]} ms</div>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

            # Show generated answer
            with st.expander("Resposta Gerada", expanded=True):
                st.markdown(
                    f'<div class="response-text">{_esc(answer)}</div>',
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"Erro na avaliacao: {e}")
