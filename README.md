# Autonomous Research Report Agent

Multi-agent system: topic -> sub-questions -> multi-source search -> verified facts -> cited report.

## Phase 1 setup (this week)

1. `pip install -r requirements.txt --break-system-packages`
2. Copy `.env.example` to `.env` and fill in your GEMINI_API_KEY and TAVILY_API_KEY
3. Run: `python main.py "your research topic"`

## Team

| Member | Owns |
|---|---|
| A | Orchestration, Planner, Critique |
| B | Search & Extraction |
| C | Verifier |
| D | Synthesizer, Eval, UI |
