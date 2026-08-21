# Autonomous Research Report Agent

An autonomous multi-agent research system that performs structured, evidence-driven research using Large Language Models.

Instead of simply searching the web and generating a summary, the system follows a research-oriented pipeline that decomposes a research topic, gathers evidence from multiple sources, evaluates sources, and produces structured research artifacts.

## 🚀 Overview

```text
Research Topic
      │
      ▼
Research Planning
      │
      ▼
Question Decomposition
      │
      ▼
Multi-Source Web Search
      │
      ▼
Source Ranking & Retrieval
      │
      ▼
Structured Evidence Extraction
      │
      ▼
Research Summary
```

The long-term goal is to build an autonomous research agent capable of producing evidence-backed, traceable, and scientifically structured reports.

---

## ✨ Features

### Phase 1 — Research Planning

The planner transforms a research topic into a structured research plan containing:

- Working hypothesis
- Competing hypotheses
- Falsification criteria
- Evaluation criteria
- Research questions
- Evidence requirements
- Source quality policies
- Search strategy
- Evidence sufficiency criteria
- Stopping criteria

**Output:**

```text
data/research_plan.json
```

### Phase 2 — Research Pipeline

The research pipeline autonomously performs:

- Multi-query web search
- Counter-evidence search
- Source deduplication
- Source ranking
- Deep content retrieval
- Structured evidence extraction
- Evidence aggregation
- Preliminary research summary

**Output:**

```text
data/research_evidence.json
```

---

## 🏗️ Architecture

```text
                     User Research Topic
                              │
                              ▼
                   ┌────────────────────┐
                   │ Phase 1            │
                   │ Research Planner   │
                   └─────────┬──────────┘
                             │
                  data/research_plan.json
                             │
                             ▼
                   ┌────────────────────┐
                   │ Phase 2            │
                   │ Research Pipeline  │
                   └─────────┬──────────┘
                             │
               data/research_evidence.json
                             │
                             ▼
                   ┌────────────────────┐
                   │ Phase 3            │
                   │ Evidence Validation│
                   │ & Reasoning        │
                   └────────────────────┘
                         Planned
```

---

## 🔬 Research Methodology

Unlike conventional RAG systems that primarily follow:

```text
Search
  ↓
Summarize
```

this project follows a research-oriented workflow:

```text
Research Topic
      ↓
Research Planning
      ↓
Question Decomposition
      ↓
Hypothesis Generation
      ↓
Multi-Source Retrieval
      ↓
Source Evaluation
      ↓
Evidence Extraction
      ↓
Evidence Synthesis
```

The system explicitly considers **competing hypotheses and counter-evidence** rather than relying on a single search-and-summarize cycle.

---

## 📊 Implementation Status

| Phase | Component | Status |
|------|-----------|--------|
| Phase 1 | Research Planning | ✅ Implemented |
| Phase 2 | Query Generation | ✅ Implemented |
| Phase 2 | Web Search & Retrieval | ✅ Implemented |
| Phase 2 | Source Ranking | ✅ Implemented |
| Phase 2 | Evidence Extraction | ✅ Implemented |
| Phase 2 | Evidence Aggregation | ✅ Implemented |
| Phase 3 | Evidence Validation | 🔄 Planned |
| Phase 3 | Contradiction Detection | 🔄 Planned |
| Phase 3 | Autonomous Re-search | 🔄 Planned |
| Phase 4 | Scientific Report Generation | 🔄 Planned |

> **Current milestone:** Phase 1 + Phase 2 initial prototype.

---

## 🧠 Current Capabilities

- Structured research planning
- Automatic research question generation
- Multi-query search strategies
- Counter-evidence retrieval
- Source deduplication and ranking
- Deep webpage retrieval
- Structured evidence extraction
- Evidence aggregation
- Preliminary research summaries
- JSON-based outputs for downstream processing

---

## 🔮 Planned Features

Future phases will introduce:

- Evidence quality enforcement
- Planner requirement validation
- Exact evidence provenance
- Page, section, and table-level citations
- Automatic evidence sufficiency checking
- Iterative autonomous re-search
- Claim-to-evidence mapping
- Contradiction detection
- Hypothesis evaluation
- Falsification checking
- Confidence scoring
- Scientific report generation

---

## 🛠️ Tech Stack

- **Python**
- **Google Gemini**
- **Tavily Search API**
- **Pydantic**
- **python-dotenv**
- **LangChain ecosystem** where applicable

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/kanishk-16/research-agent.git
cd research-agent
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

> Never commit your `.env` file or expose API keys publicly.

---

## ▶️ Usage

Run the application:

```bash
python main.py
```

Enter a research topic when prompted.

### Example

```text
Is Retrieval-Augmented Generation actually effective at reducing hallucinations in LLMs?
```

Other example topics:

- Does Chain-of-Thought reasoning improve LLM accuracy?
- Does long-context eliminate the need for RAG?
- Are AI coding assistants improving developer productivity?

---

## 📁 Output

The system generates structured research artifacts:

```text
data/
├── research_plan.json
└── research_evidence.json
```

The application also produces a console-based research summary.

---

## 📂 Repository Structure

```text
research-agent/
│
├── agents/
│   └── ...
│
├── planner/
│   └── ...
│
├── search_agent/
│   ├── content_retriever.py
│   ├── evidence_extractor.py
│   ├── evidence_output.py
│   ├── query_generator.py
│   ├── researcher.py
│   └── source_ranker.py
│
├── data/
│   ├── research_plan.json
│   └── research_evidence.json
│
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## 👥 Team

| Member | Responsibility |
|--------|----------------|
| **Yash Thakur** | Research Planner — Phase 1 |
| **Sumit Kumar** | Research Pipeline & Evidence Extraction — Phase 2 |
| **Kanishk Vikram Singh** | Evidence Validation & Reasoning — Phase 3 |
| **Devansh Saini** | Report Generation, Evaluation & UI — Phase 4 |

---

## 📌 Project Status

**Current:** Phase 1 + Phase 2 initial prototype

**Next:** Evidence validation, autonomous research refinement, and scientific report generation.

This project is being developed as part of an academic research project.

---

## 📄 License

This project is developed for academic research and educational purposes.
