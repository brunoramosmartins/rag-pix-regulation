.PHONY: help test test-fast lint format weaviate phoenix ingest chunk index pipeline demo eval-retrieval eval-rag

PYTHON := python
PYTEST := pytest

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "RAG Pix Regulation — Development Commands"
	@echo ""
	@echo "  Setup"
	@echo "    make weaviate          Start Weaviate via Docker Compose"
	@echo "    make phoenix           Start Phoenix observability server"
	@echo ""
	@echo "  Pipeline"
	@echo "    make ingest            PDF → corpus_pages.jsonl"
	@echo "    make chunk             Pages → corpus_chunks.jsonl"
	@echo "    make index             Chunks → embeddings → Weaviate"
	@echo "    make pipeline          Full pipeline (ingest + chunk + init + index)"
	@echo "    make demo              Launch Streamlit demo"
	@echo ""
	@echo "  Evaluation"
	@echo "    make eval-retrieval    Evaluate retrieval metrics (Precision@K, Recall@K)"
	@echo "    make eval-rag          Evaluate full RAG pipeline (requires Ollama)"
	@echo ""
	@echo "  Quality"
	@echo "    make test              Run unit tests (excludes slow/integration tests)"
	@echo "    make test-all          Run all tests"
	@echo "    make lint              Check code style with ruff"
	@echo "    make format            Auto-format code with ruff"

# ── Infrastructure ────────────────────────────────────────────────────────────

weaviate:
	docker compose up -d

phoenix:
	$(PYTHON) -m phoenix.server.main serve

# ── Data Pipeline ─────────────────────────────────────────────────────────────

ingest:
	$(PYTHON) scripts/run_ingestion.py

chunk:
	$(PYTHON) scripts/run_chunking.py

index:
	$(PYTHON) scripts/run_indexing.py

pipeline:
	$(PYTHON) scripts/run_pipeline.py

demo:
	streamlit run app/streamlit_app.py

# ── Evaluation ────────────────────────────────────────────────────────────────

eval-retrieval:
	$(PYTHON) scripts/evaluate_retrieval.py

eval-rag:
	$(PYTHON) scripts/evaluate_rag.py

# ── Quality ───────────────────────────────────────────────────────────────────

test:
	$(PYTEST) -m "not slow and not integration" -v

test-all:
	$(PYTEST) -v

lint:
	ruff check .

format:
	ruff format .
