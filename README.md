# ARA — Autonomous Research Agent

> **Evidence-driven research from question to conclusion**

ARA (Autonomous Research Agent) is a multi-agent autonomous research system designed for rigorous, evidence-backed scientific and technical inquiry using Large Language Models.

Unlike conventional search-and-summarize RAG pipelines that produce shallow overviews, ARA operates like a scientific investigator: decomposing inquiries into testable hypotheses, actively seeking counter-evidence, querying multi-provider academic databases (OpenAlex, Semantic Scholar) and web sources, enforcing strict evidence provenance with verbatim quotes, and synthesizing findings into an auditable academic report with deterministic citations.

```
     █████╗  ██████╗   █████╗ 
    ██╔══██╗ ██╔══██╗ ██╔══██╗
    ███████║ ██████╔╝ ███████║
    ██╔══██║ ██╔══██╗ ██╔══██║
    ██║  ██║ ██║  ██║ ██║  ██║
    ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝

       AUTONOMOUS RESEARCH AGENT
   Evidence-driven research from question to conclusion
```

---

## 🌟 Key Capabilities

- **Modern Developer CLI**: Rich-powered terminal interface featuring an original visual identity, live status bars, subtle spinners, debounced provider notifications, quantitative metrics cards, and compact source tables.
- **Zero-Shot Vector Similarity Gating**: Domain-agnostic relevance ranking that strips universal methodology boilerplate and compares query/question subject intent against candidate abstracts via L2-normalized subject vector cosine similarity and focal entity coverage — supporting AI/CS, life sciences, biology, medicine, and chemistry inquiries without hardcoded field blacklists.
- **Pydantic Structured Evidence Extraction**: Token-level constrained decoding via Gemini SDK `response_schema=FindingExtractionModel` (`ExtractedFindingItemModel`, `QuantitativeEvidenceModel`, `EvidenceExcerptModel`) with direct structured LLM stance classification (`support`, `counter`, `limitation`, `trade_off`, `mixed`, `context`).
- **Asynchronous Search & Parallel PDF Pipeline (`asyncio.gather`)**: Concurrently queries **Tavily**, **OpenAlex**, **Semantic Scholar**, and **Europe PMC** across all sub-questions via `asyncio.gather()`, downloads and parses candidate PDFs in parallel via a thread pool, and handles rate limits with non-blocking jittered exponential backoff.
- **6-Step Information Retrieval Engine**: Pure-Python NLP pipeline featuring stop-word removal, 5-stage Porter Stemming, inverted indexing with Robertson–Spärck Jones BM25 scoring, dense vector embeddings (`gemini-embedding-001`), cosine similarity, and prompt-aligned candidate re-ranking.
- **Two-Stage Candidate Filtering & 350-Word Passage Chunking**: Pre-filters raw search candidates down to the top 35 via fast BM25 before generating dense vector embeddings, and segments full-text papers into 350-word overlapping passages to prevent *"Lost in the Middle"* context degradation.
- **Academic API & Full-Text Resolution Suite**:
  - **OpenAlex & Dual-Mode Semantic Scholar**: Graph API + autonomous web discovery fallback to prevent HTTP 429 blocks.
  - **ar5iv / Direct arXiv HTML5**: Fetches pre-rendered HTML5 (`ar5iv.labs.arxiv.org/html/<id>` and `arxiv.org/html/<id>`) to preserve tables and formulas before falling back to binary PDF parsing.
  - **Unpaywall DOI Open-Access Resolver**: Resolves direct open-access legal PDF URLs from any DOI (`api.unpaywall.org/v2/<doi>`).
  - **Europe PMC / PubMed REST API**: Structured biomedical and life-sciences full-text and abstract retrieval (`EuropePMCProvider`).
- **5-Gate Evidence Validation Layer (EVL)**: Rigorously verifies evidence across 5 strict gates: target variable alignment, character-for-character quote provenance against raw text, quantitative metric grounding with auto-calculated differences, adversarial counter-evidence calibration, and source independence clustering.
- **Relational + Vector Database, Multi-Tier Caching & 3-Phase Checkpointing**:
  - Relational schemas (`users`, `projects`, `runs`, `sources`, `findings`) + 350-word `passages` vector column storage (SQLite + PostgreSQL `pgvector` HNSW DDL).
  - Multi-tier SHA-256 cache for Search Queries (21-day TTL), Parsed PDFs, and Passage Embeddings.
  - Automatic state checkpointing after Phase 1 (Plan), Phase 2 (Search/Passages), and Phase 3 (Evidence) for instant crash resumption.
- **Authentication, Multi-Tenancy, BYOK Encryption & Production Stack**:
  - HS256 JWT authentication, multi-tenant organization workspaces with monthly query quotas, encrypted BYOK API keys (`cryptography.fernet` / AEAD), and `python main.py login` CLI OAuth handshake.
  - Unified `ResearchOrchestrator` service powering both the CLI and web backends, background task queue (`BackgroundTaskQueue`), Server-Sent Events (SSE) live progress streaming (`/api/stream`), FastAPI OpenAPI backend, Jinja2 HTML report template (`templates/report.html.jinja2`), and Docker Compose stack (`api`, `worker`, `redis`, `postgres`).
- **Deterministic Offline Demo (`--demo`)**: Full offline rehearsal mode that exercises the entire CLI UI, live spinners, metric summaries, and HTML report generation with zero API keys and zero external network calls.

---

## 🏗️ Architecture & Pipeline

```text
                        User Research Topic / Hypothesis
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: RESEARCH PLANNING                                                  │
│ • Hypothesis generation (working & competing)                               │
│ • Falsification criteria & evaluation methodology                           │
│ • Question decomposition & evidence requirements                            │
│ ➔ Emits: data/research_plan.json                                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: MULTI-PROVIDER SEARCH, SEMANTIC INDEXING & PASSAGE CHUNKING        │
│ • Multi-angle query formulation (targeted, mechanism, counter-evidence)     │
│ • Tri-provider discovery (OpenAlex + Semantic Scholar + Tavily)             │
│ • Multi-key deduplication (DOI, arXiv ID, URL, Title)                       │
│ • 6-Step Semantic Indexing: BM25 inverted index + Gemini dense embeddings   │
│ • Two-Stage Filtering: BM25 coarse (500 -> 35) -> Dense vector (-> 10)      │
│ • Sliding-window PDF passage chunking (350 words, 50-word overlap)          │
│ ➔ Emits: data/research_evidence.json                                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: EVIDENCE EXTRACTION & VALIDATION                                    │
│ • Extraction of verbatim quotes, quantitative findings, confidence scores   │
│ • Strict validation against source text                                     │
│ • Counter-evidence calibration & cross-question routing                     │
│ • Sufficiency evaluation (triggers autonomous re-search loop if unmet)      │
│ ➔ Emits: data/research_evidence.json                                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: SYNTHESIS & REPORT GENERATION                                      │
│ • Synthesis across hypotheses, consensus, nuances, and contradictions       │
│ • Deterministic citation assignment [1], [2], ...                           │
│ • Multi-format artifact generation:                                         │
│   ├─ data/research_report.html (Primary academic report)                    │
│   ├─ data/research_report.md   (Markdown document)                          │
│   └─ data/research_report.json (Machine-readable report graph)              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- Virtual environment recommended

### 2. Installation

```bash
git clone https://github.com/kanishk-16/research-agent.git
cd research-agent

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file in the project root:

```env
# Required for LLM planning, extraction, and synthesis
GEMINI_API_KEY=your_gemini_api_key

# Required for web search & academic discovery
TAVILY_API_KEY=your_tavily_api_key

# Optional: Semantic Scholar API Key (elevates rate limits)
SEMANTIC_SCHOLAR_API_KEY=your_s2_api_key
```

> **Note:** ARA operates gracefully even without `SEMANTIC_SCHOLAR_API_KEY`, using autonomous web discovery fallback if rate limits are encountered.

---

## 💻 CLI Usage

### Interactive Research Mode

Launch ARA with its interactive developer prompt:

```bash
python main.py
```

You will see the startup banner, provider status, and an interactive prompt:

```text
ARA › Under what conditions does retrieval-augmented generation reduce hallucination in large language models?
```

### Direct Topic Execution

Pass the research question directly as an argument:

```bash
python main.py "Under what conditions does retrieval-augmented generation reduce hallucination in large language models, and when can retrieval actually degrade factual accuracy?"
```

### Interactive Semantic Pipeline Demo (`demo_semantic_pipeline.py`)

Run an interactive, step-by-step console visualization of the entire 7-step semantic indexing, BM25 inverted index, vector cosine similarity, and PDF passage chunking pipeline:

```bash
python demo_semantic_pipeline.py
```

### Offline Demo Mode (`--demo`)

Rehearse the complete CLI UI and generate full research artifacts offline without external API keys or network requests:

```bash
python main.py --demo
```

The demo mode:
1. Simulates realistic planning, discovery, evidence extraction, and synthesis.
2. Emits full artifacts to `data/`.
3. Automatically opens `data/research_report.html` in your default web browser upon completion.

### Verbose Mode (`--verbose`)

Enable detailed internal logging and diagnostic outputs:

```bash
python main.py --verbose
```

### CLI OAuth Login Handshake (`login` / `--login`)

Authenticate a local terminal session and persist signed JWT credentials to `~/.ara/credentials.json`:

```bash
python main.py login
```

### Containerized Production Stack (Docker Compose)

Launch the full production stack (`api`, `worker`, `redis`, and `postgres` with `pgvector`):

```bash
docker compose up --build -d
```

---

## 📄 Research Artifacts

Every research run produces five synchronized artifacts in `data/`:

| Artifact | Format | Description |
|---|---|---|
| [`data/research_report.html`](file:///d:/research-agent/data/research_report.html) | HTML5 | **Primary human artifact**. Rendered via Jinja2 (`templates/report.html.jinja2`) with deterministic numerical citations `[1]`, canonical DOI/arXiv links, quantitative finding tables, and counter-evidence callouts. |
| [`data/research_report.md`](file:///d:/research-agent/data/research_report.md) | Markdown | Portable markdown synthesis report with bibliography. |
| [`data/research_report.json`](file:///d:/research-agent/data/research_report.json) | JSON | Machine-readable report graph with metadata, executive summary, and claim-evidence mappings. |
| [`data/research_evidence.json`](file:///d:/research-agent/data/research_evidence.json) | JSON | Granular evidence repository with verbatim quotes, confidence scores, and methodology ratings. |
| [`data/research_plan.json`](file:///d:/research-agent/data/research_plan.json) | JSON | Decomposed research plan, competing hypotheses, falsification criteria, and sufficiency quotas. |

---

## 🧪 Testing

ARA includes a comprehensive test suite (**190 passing unit tests**) covering CLI UI rendering, Jinja2 HTML report generation, offline demo execution, zero-shot cross-domain ranking, Pydantic `FindingExtractionModel` constrained decoding, `asyncio.gather()` multi-provider concurrency, jittered exponential backoff, multi-tier SHA-256 caching, 3-phase checkpointing, relational/vector database search, ar5iv/Unpaywall/Europe PMC integrations, JWT/BYOK authentication, SSE streaming, and 5-gate EVL validation:

```bash
# Run entire test suite (190 tests)
pytest

# Run semantic indexing & chunking tests specifically
pytest tests/test_semantic_indexer.py -v

# Run Evidence Validation Layer (EVL) tests
pytest tests/test_evidence_validation_layer.py -v

# Run generalized pipeline, Pydantic extraction, async, DB/cache & auth tests
pytest tests/test_generalized_pipeline.py -v

# Run CLI and report tests specifically
pytest tests/test_cli_ui.py tests/test_html_report_and_demo.py -v
```

---

## 📁 Repository Structure

```text
research-agent/
│
├── cli/                        # Terminal CLI UI (Rich-based)
│   ├── console.py              # ARAConsole coordinator & state management
│   ├── components.py           # Banners, cards, source tables, and metrics
│   ├── progress.py             # Live spinner and progress bars
│   └── theme.py                # Color palette and typographic styles
│
├── demo/                       # Offline demonstration engine
│   ├── demo_data.py            # Representative academic dataset & report
│   └── demo_runner.py          # Zero-network simulation runner
│
├── templates/                  # Jinja2 HTML/CSS Templates
│   └── report.html.jinja2      # Self-contained academic HTML report template
│
├── reports/                    # Report renderers
│   └── html_renderer.py        # Jinja2-backed academic HTML report generator
│
├── planner/                    # Phase 1: Research Planning
│   ├── planner.py              # Multi-agent question decomposition
│   └── validation/             # Plan validation, quality, and contract enforcer
│
├── critique/                   # Plan quality critique & falsification review
│   └── plan_critic.py          # Evaluates research plan validity and counter-evidence quotas
│
├── search_agent/               # Phase 2 & 3: Retrieval, Semantic Indexing & Evidence Engine
│   ├── ranking/                # Modularized source classification, policy & zero-shot scoring
│   │   ├── policy.py           # Source type classification, quality tiers & eligibility
│   │   └── scorer.py           # Zero-shot subject vector similarity & ranking diagnostics
│   ├── extraction/             # Modularized Pydantic schemas & grounding verification
│   │   ├── models.py           # FindingExtractionModel & Pydantic validators
│   │   └── grounding.py        # Quantitative number grounding & provenance inference
│   ├── orchestrator.py         # Unified ResearchOrchestrator service for CLI & Web UI
│   ├── researcher.py           # Asynchronous asyncio.gather() multi-provider search pipeline
│   ├── source_providers.py     # OpenAlex, Semantic Scholar (Dual-Mode) & Europe PMC REST APIs
│   ├── content_retriever.py    # Parallel PDF parser, ar5iv HTML5 & Unpaywall DOI OA resolver
│   ├── semantic_indexer.py     # 6-step IR pipeline, Porter stemmer, BM25 & PDF passage chunker
│   ├── source_ranker.py        # Public facade for search_agent.ranking
│   ├── evidence_extractor.py   # Constrained decoding (response_schema) & multi-question routing
│   ├── evidence_validator.py   # 5-Gate EVL validation & character-level quote provenance
│   ├── evidence_sufficiency.py # Sufficiency evaluation & re-search triggers
│   ├── database.py             # Relational + 350-word passage vector storage (SQLite / pgvector)
│   ├── cache_manager.py        # Multi-tier SHA-256 cache (Search Queries, PDFs, Embeddings)
│   ├── checkpoint_manager.py   # 3-phase state checkpointing & crash resumption
│   ├── auth_manager.py         # JWT auth, multi-tenant quotas, BYOK encryption & CLI login
│   ├── task_queue.py           # Background job worker queue & Server-Sent Events (SSE) formatter
│   └── tee_logger.py           # Diagnostic stdout capture
│
├── verifier/                   # Fail-closed claim audit engine
│   └── claim_verifier.py       # Verifies quotes and claims against source text
│
├── synthesizer/                # Phase 4: Calibrated Report Generation
│   └── calibrated_synthesizer.py # Epistemic Confidence Score & calibrated synthesis
│
├── ui/                         # Web Dashboards & REST/SSE APIs
│   ├── server.py               # Web server with unified ResearchOrchestrator & /api/stream SSE
│   └── fastapi_app.py          # Asynchronous FastAPI backend with OpenAPI & background jobs
│
├── data/                       # Generated research artifacts & checkpoints
├── tests/                      # Automated test suite (190 tests passing)
├── Dockerfile                  # Production container image definition
├── docker-compose.yml          # Multi-container stack (api, worker, redis, postgres/pgvector)
├── demo_semantic_pipeline.py   # Interactive 7-step IR & passage chunking console demo
├── main.py                     # CLI entrypoint
└── requirements.txt            # Project dependencies
```

---

## 👥 Team

| Member | Focus Area |
|---|---|
| **Yash Thakur** | Research Planning & Problem Decomposition (Phase 1) |
| **Sumit Kumar** | Search Pipeline & Evidence Extraction (Phase 2) |
| **Kanishk Vikram Singh** | Evidence Validation, Epistemics & Re-search (Phase 3) |
| **Devansh Saini** | Synthesis, Evaluation, CLI UI & HTML Reports (Phase 4) |

---

## 📄 License

This project is developed for academic research, benchmarking, and educational purposes.
