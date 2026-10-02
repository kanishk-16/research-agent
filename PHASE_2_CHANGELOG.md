# Generalized Research Pipeline Overhaul: Extraction → Alignment → Independence → Validation → Synthesis

This document provides a comprehensive technical overview of the generalized architectural overhaul of the Autonomous Academic Research Agent (AARA), addressing 15 core vulnerabilities across the entire research lifecycle, plus dual-mode Semantic Scholar integration.

---

## 1. Executive Summary & 5-Stage Architecture

The research pipeline has been upgraded into a modular 5-stage pipeline:

```text
 1. EXTRACTION          2. ALIGNMENT          3. INDEPENDENCE        4. VALIDATION          5. SYNTHESIS
┌───────────────┐      ┌───────────────┐      ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ Dual-Mode S2  │ ───► │ Target Var    │ ───► │ Finding Merge │ ───► │ Contract      │ ───► │ Calibrated    │
│ OpenAlex +    │      │ Gating +      │      │ Dedup + Max 2 │      │ Enforcer +    │      │ Confidence    │
│ Tavily        │      │ Archetypes    │      │ Cap per Paper │      │ EVL 5 Gates   │      │ Report (.md)  │
└───────────────┘      └───────────────┘      └───────────────┘      └───────────────┘      └───────────────┘
```

The system is tested and verified by **69 unit tests** passing with 100% success in **~0.03 seconds**.

---

## 2. The 15 Generalized Problems & Solutions

| Area | Problem Addressed | Technical Solution Implemented |
|---|---|---|
| **1. Extraction** | **P1: Evidence extraction reliability**<br/>Relevant sources yield little/no usable evidence. | **Multi-Pass Extraction**: Extracts findings from abstracts, high-density empirical sections (Results, Evaluation, Discussion), and fallback empirical sentence matching. |
| | **P7: Source utilization efficiency**<br/>Selected sources yield zero findings. | **Pre-Selection Extractability Density Scoring**: [`_calculate_extractability_density()`](search_agent/source_selector.py) scores candidate text before allocating selection slots. Tracks `source_yield_rate`. |
| | **P10: Search-to-evidence conversion**<br/>Conversion bottleneck downstream of discovery. | **Funnel Analytics**: Tracks conversion rates across `discovered -> selected -> retrieved -> extracted -> validated`. |
| **2. Alignment** | **P2: Evidence-to-question alignment**<br/>Evidence assigned to questions it does not answer. | **Gate 1 (Target Variable Alignment)**: Strictly matches findings to sub-question target variables (`accuracy`, `latency`, `cost`, `failure_modes`, `boundary`). Restricts global cross-routing. |
| | **P9: Adaptive research planning**<br/>Requirements poorly matched to question types. | **Question-Type Profiling**: Calibrates requirements adaptively based on question archetype (`empirical_comparison`, `mechanistic_design`, `theoretical_limit`, `definitional_context`). |
| | **P13: Planner–Researcher contract enforcement**<br/>Requirements not verified in final evidence. | **Contract Enforcement Engine** ([`planner/validation/contract_enforcer.py`](planner/validation/contract_enforcer.py)): Formally validates plan quotas (sources, primary, quantitative, counter) against extracted evidence. |
| **3. Independence & Diversity** | **P3: Duplicate evidence**<br/>Semantically equivalent findings inflate counts. | **Finding-Level Semantic Deduplication**: [`_is_semantically_equivalent_claim()`](search_agent/evidence_extractor.py) merges redundant findings into canonical records with multi-source attribution (`source_ids: ["S1", "S2"]`). |
| | **P4: Evidence concentration**<br/>Findings concentrated in 1–2 sources. | **Per-Source Finding Cap**: Caps maximum findings per unique source per question (default 2), preventing single papers from dominating the evidence. Tracks HHI concentration ratio. |
| | **P6: Quantitative evidence independence**<br/>Multiple numbers from one publication inflate evidence. | **Independent Quantitative Paper Counting**: Sufficiency requires distinct quantitative *papers*, not raw numbers. |
| | **P11: Evidence diversity**<br/>Missing diversity across authors, institutions, benchmarks, time. | **Multi-Dimensional Diversity Tracker**: Enforces venue diversity, author caps, benchmark diversity (e.g. GSM8K, HumanEval), and temporal spans. |
| **4. Validation** | **P5: Counter-evidence coverage**<br/>System biased towards supporting evidence. | **Adversarial Retrieval Pass**: Formulates counter queries probing falsification criteria, baseline parity, failure modes, and competing hypotheses. |
| | **P8: Source identity & provenance validation**<br/>Cross-referencing inconsistencies. | **Cryptographic Provenance Tuple**: `(canonical_id, doi, canonical_url, title_hash)` attached to every source and finding with bidirectional reference integrity checks. |
| | **P12: Contradiction detection dependency**<br/>Contradiction engine idle without opposing evidence. | **Active Falsification Retrieval**: When zero counter-findings are extracted, automatically triggers targeted searches specifically probing falsification boundaries. |
| | **P14: Evidence-quality calibration**<br/>Conflating findings, sources, and independent sources. | **Epistemic Triplet**: Always reports `[finding_count, source_count, independent_source_count]` across all pipeline summaries and JSON outputs. |
| **5. Synthesis** | **P15: Final synthesis calibration**<br/>Tone & confidence mismatching evidence strength. | **Phase 3 Calibrated Synthesizer** ([`synthesizer/calibrated_synthesizer.py`](synthesizer/calibrated_synthesizer.py)): Calculates mathematical Epistemic Confidence Score ($0.0 - 1.0$) driving structured report generation with calibrated hedging tiers (`HIGH`, `MODERATE`, `LOW_HEDGED`). |

---

## 3. Dual-Mode Semantic Scholar Integration

To resolve the Semantic Scholar HTTP 429 barrier while fulfilling the requirement to include Semantic Scholar:

* **Mode A (Authenticated)**: Uses the Semantic Scholar Graph API (`api.semanticscholar.org`) when `SEMANTIC_SCHOLAR_API_KEY` is present in `.env`.
* **Mode B (Autonomous Web Discovery)**: When no API key is set, discovers authentic Semantic Scholar papers via Tavily (`site:semanticscholar.org/paper <query>`). Extracts Semantic Scholar Paper IDs, canonical URLs, DOIs, and abstracts without encountering HTTP 429 blocks.

```text
[Academic Providers] OpenAlex: Active | Semantic Scholar: Active (Dual-Mode)
```

---

## 4. File-by-File Summary of Additions & Changes

### New Modules
| File | Description |
|---|---|
| [`planner/validation/contract_enforcer.py`](planner/validation/contract_enforcer.py) | Evaluates compliance between research plan criteria and researcher evidence. |
| [`synthesizer/calibrated_synthesizer.py`](synthesizer/calibrated_synthesizer.py) | Generates calibrated final research reports with Epistemic Confidence scoring and structured sections. |
| [`tests/test_generalized_pipeline.py`](tests/test_generalized_pipeline.py) | Unit tests covering Semantic Scholar Dual-Mode, finding deduplication, per-source caps, density scoring, and contract enforcement. |

### Enhanced Modules
| File | Improvements Made |
|---|---|
| [`search_agent/source_providers.py`](search_agent/source_providers.py) | Added Dual-Mode `SemanticScholarProvider` (Mode A: Graph API, Mode B: Tavily Web Discovery). |
| [`search_agent/researcher.py`](search_agent/researcher.py) | Wired Dual-Mode Semantic Scholar into primary and targeted search passes alongside OpenAlex. |
| [`search_agent/content_retriever.py`](search_agent/content_retriever.py) | Full-text engine: added arXiv HTML/PDF resolution, native pypdf binary extraction, and accurate empirical content status scoring. |
| [`search_agent/evidence_validator.py`](search_agent/evidence_validator.py) | Enhanced EVL: added morphological stemmer, defense taxonomies in Gate 1, citation-resilient token-containment in Gate 2, qualitative preservation in Gate 3, and security counter patterns in Gate 4. |
| [`search_agent/evidence_extractor.py`](search_agent/evidence_extractor.py) | Added archetype-gated cross-routing (`_is_cross_routing_allowed`), defense prompt rules, and acronym-aware deduplication. |
| [`search_agent/source_selector.py`](search_agent/source_selector.py) | Added pre-selection extractability density scoring to prioritize rich empirical sources. |
| [`search_agent/evidence_sufficiency.py`](search_agent/evidence_sufficiency.py) | Added `independent_source_count` and `epistemic_triplet` to evaluation results. |
| [`search_agent/evidence_output.py`](search_agent/evidence_output.py) | Added Epistemic Triplet, source yield rate, and contract enforcement matrix to output JSON. |
| [`planner/validation/contract_enforcer.py`](planner/validation/contract_enforcer.py) | Primary source counting: recognizes `conference_paper`, preprints, DOIs, and academic provider metadata. |
| [`planner/validation/quality_validator.py`](planner/validation/quality_validator.py) | Intelligent question distinctness: disambiguates parallel comparative questions across distinct domains/tasks (e.g. math vs code) from true duplicate questions. |
| [`search_agent/source_providers.py`](search_agent/source_providers.py) | Semantic Scholar Web Discovery: strips `/figure/` and `/table/` sub-pages from URLs and titles to target canonical papers. |
| [`search_agent/content_retriever.py`](search_agent/content_retriever.py) | Multi-path fallback retrieval with title-based academic resolution and snippet-only reconciliation preventing failed status. |
| [`search_agent/source_ranker.py`](search_agent/source_ranker.py) | Domain mismatch filtering with expanded electrophysiology/cardiology keywords (`endocardial`, `myocardial`, `arrhythmia`, `atrial fibrillation`, etc.) and enhanced `_is_ai_query` detecting decoding-time interventions, activation steering, and knowledge conflict resolution. |
| [`search_agent/evidence_extractor.py`](search_agent/evidence_extractor.py) | Multi-pass extraction, density-guided ranking, heuristic quantitative metric parser (including rates, proportions, and thresholds), cross-routing quantitative enrichment with refined trade-off/failure gating (disentangled from hardware bottlenecks), transient Gemini API 503/500 retry with exponential backoff, and expanded counter-evidence patterns (deterioration, performance drops, degradation below baseline). |
| [`search_agent/evidence_validator.py`](search_agent/evidence_validator.py) | EVL 5 Gates, Gate 3 standalone overhead and boundary threshold support, expanded cost/latency metric keywords, and Gate 4 claim-level counter prioritization with performance degradation calibration. |
| [`search_agent/evidence_sufficiency.py`](search_agent/evidence_sufficiency.py) | Harmonized primary-source and minimum source counting across global selection bundle and cross-routed findings using `effective_source_count`, fully aligning with contract enforcement. |
| [`main.py`](main.py) | Wired Phase 1 Planner → Phase 2 Researcher (with Dual-Mode S2) → Phase 3 Calibrated Synthesizer. |

---

## 5. Verification & Automated Test Suite

All **91 unit tests** pass with 100% success:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
# Ran 91 tests in ~7s - OK
```

### Module Breakdown:
* `tests/test_live_feedback_fixes.py`: **22 passed** (including latency overhead extraction, parallel question disambiguation, counter-evidence baseline calibration, retrieval fallback reconciliation, speculative decoding speedup failure calibration, proportion and acceptance threshold extraction, Gate 3 threshold acceptance, transient 503 retry, knowledge conflict performance drop calibration, endocardial biomedical disqualification from AI questions, and primary source counting with cross-routed findings)
* `tests/test_generalized_pipeline.py`: **7 passed**
* `tests/test_evidence_validation_layer.py`: **16 passed**
* `tests/test_end_to_end_research_eval.py`: **28 passed**
* `tests/test_phase2_completion.py`: **18 passed**
* **Total: 91 passed, 0 failures, 0 errors**

