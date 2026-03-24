"""CSS styles for the Streamlit application — light professional theme."""

CSS = """
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

  /* ── Diagnostics panel ── */
  .diag-section {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-top: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .diag-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #2563eb;
    margin-bottom: 0.8rem;
    font-weight: 600;
  }
  .diag-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.75rem;
  }
  .diag-item {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
  }
  .diag-item-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.56rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.2rem;
  }
  .diag-item-value {
    font-size: 1rem;
    font-weight: 600;
    color: #0f172a;
  }
  .score-bar {
    height: 6px;
    background: #e2e8f0;
    border-radius: 3px;
    margin-top: 0.3rem;
    overflow: hidden;
  }
  .score-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  /* ── Evaluation panel ── */
  .eval-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .eval-query {
    font-size: 0.88rem;
    color: #1e293b;
    font-weight: 500;
    margin-bottom: 0.5rem;
  }
  .eval-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.56rem;
    padding: 0.15rem 0.5rem;
    border-radius: 100px;
    font-weight: 500;
    letter-spacing: 0.06em;
  }
  .eval-badge.single_chunk  { background: #dbeafe; color: #1d4ed8; }
  .eval-badge.multi_chunk   { background: #fef3c7; color: #d97706; }
  .eval-badge.cross_section { background: #fce7f3; color: #be185d; }
  .eval-badge.negative      { background: #fee2e2; color: #dc2626; }

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
"""
