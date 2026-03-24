"""Query input section — text area, example selector, and run button."""

import streamlit as st

EXAMPLE_QUERIES = [
    "Como funciona o registro de chave Pix?",
    "Quais sao as regras de devolucao por fraude?",
    "Como funciona a portabilidade de chave Pix?",
    "Quantas chaves Pix posso ter por conta?",
    "O que e o DICT no contexto do Pix?",
    "Quais sao os estados possiveis de uma notificacao de infracao no DICT?",
    "Qual e o prazo para iniciar uma solicitacao de devolucao apos fraude?",
]


def render_query_input() -> tuple[str, bool]:
    """Render query input area and return (query_text, run_clicked)."""
    st.markdown('<div class="query-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="query-label">Consulta</div>', unsafe_allow_html=True)

    query = st.text_area(
        label="",
        placeholder="Digite sua pergunta sobre regulamentacao Pix...",
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
        options=["Selecione um exemplo..."] + EXAMPLE_QUERIES,
        key="example_select",
        label_visibility="collapsed",
    )
    actual_query = query.strip() or (
        example if example and example != "Selecione um exemplo..." else ""
    )

    st.markdown("</div>", unsafe_allow_html=True)

    run_clicked = st.button("Executar comparacao", use_container_width=True)

    return actual_query, run_clicked
