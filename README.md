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
- **Academic Citation-Rich HTML Report**: Generates `data/research_report.html` as the primary human-facing artifact, complete with numbered citations `[1]`, bidirectional footnote jumps, canonical DOI/arXiv links, quantitative finding tables, and counter-evidence callouts.
- **Deterministic Offline Demo (`--demo`)**: Full offline rehearsal mode that exercises the entire CLI UI, live spinners, metric summaries, and HTML report generation with zero API keys and zero external network calls.
- **Multi-Provider Discovery**: Interleaves academic databases (OpenAlex, Semantic Scholar) with deep web research (Tavily), featuring autonomous fallback and rate-limit backoff.
- **Multi-Key Deduplication**: Normalizes sources across digital object identifiers (DOI), arXiv IDs, normalized URL canonicalization, and fuzzy title matching.
- **Deep Content Retrieval**: Extracts full text from arXiv abstracts, arXiv full-text PDFs, Open Access publisher articles, and web pages.
- **Strict Evidence Extraction & Validation**: Extracts verifiable evidence cards containing exact quotes, numerical metrics, sample sizes, and methodology tiers. Discards unsupported claims and validates falsification criteria.
- **Autonomous Sufficiency & Re-Search**: Checks whether working hypotheses, counter-evidence quotas, and minimum evidence thresholds are satisfied, triggering targeted re-search loops when gaps exist.

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
│ PHASE 2: MULTI-PROVIDER SEARCH & DISCOVERY                                  │
│ • Query formulation (targeted + counter-evidence queries)                   │
│ • OpenAlex, Semantic Scholar & Tavily discovery                             │
│ • Multi-key deduplication (DOI, arXiv ID, URL, Title)                       │
│ • Multi-dimensional ranking (Academic peer review, recency, relevance)       │
│ • Deep text retrieval (arXiv PDFs, publisher articles, web scrape)          │
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

---

## 📄 Research Artifacts

Every research run produces five synchronized artifacts in `data/`:

| Artifact | Format | Description |
|---|---|---|
| [`data/research_report.html`](file:///d:/research-agent/data/research_report.html) | HTML5 | **Primary human artifact**. Polished, standalone academic layout, deterministic numerical citations `[1]`, canonical DOI/arXiv links, quantitative finding tables, and counter-evidence callouts. |
| [`data/research_report.md`](file:///d:/research-agent/data/research_report.md) | Markdown | Portable markdown synthesis report with bibliography. |
| [`data/research_report.json`](file:///d:/research-agent/data/research_report.json) | JSON | Machine-readable report graph with metadata, executive summary, and claim-evidence mappings. |
| [`data/research_evidence.json`](file:///d:/research-agent/data/research_evidence.json) | JSON | Granular evidence repository with verbatim quotes, confidence scores, and methodology ratings. |
| [`data/research_plan.json`](file:///d:/research-agent/data/research_plan.json) | JSON | Decomposed research plan, competing hypotheses, falsification criteria, and sufficiency quotas. |

---

## 🧪 Testing

ARA includes a comprehensive test suite (163 tests) covering CLI UI rendering, HTML generation, offline demo execution, deduplication, ranking algorithms, evidence validation, and provider fallback resilience:

```bash
# Run entire test suite
pytest

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
├── reports/                    # Report renderers
│   └── html_renderer.py        # Self-contained academic HTML generator
│
├── planner/                    # Phase 1: Research Planning
│   ├── planner.py              # Multi-agent question decomposition
│   └── validation/             # Plan validation and schema checks
│
├── search_agent/               # Phases 2 & 3: Retrieval & Evidence Engine
│   ├── source_providers.py     # OpenAlex, Semantic Scholar & Tavily connectors
│   ├── content_retriever.py    # Deep PDF & text retrieval
│   ├── source_ranker.py        # Multi-factor academic ranking
│   ├── evidence_extractor.py   # Verbatim quote & quantitative extraction
│   ├── evidence_validator.py   # Claim validation & falsification checking
│   ├── evidence_sufficiency.py # Sufficiency evaluation & re-search triggers
│   ├── researcher.py           # Pipeline orchestrator
│   └── tee_logger.py           # Diagnostic stdout capture
│
├── ui/                         # Web dashboard (FastAPI)
│   └── server.py               # REST API & live inspection UI
│
├── data/                       # Generated research artifacts
├── tests/                      # Automated test suite (163 tests)
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
