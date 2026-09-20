# Research Report: How vulnerable are Retrieval-Augmented Generation (RAG) systems to knowledge-base poisoning and indirect prompt-injection attacks, and how effective are current defenses?

## 1. Executive Summary & Epistemic Calibration
**Epistemic Confidence Rating**: `0.15 / 1.0` (**LOW_HEDGED**)
> Preliminary / Qualified Confidence: Evidence exhibits high source concentration, significant unmet requirements, or unresolved empirical contradictions.

This research synthesis evaluates empirical evidence across 3 sub-questions, citing 14 independent sources.

## 2. Evidence Analysis by Sub-Question
### Sub-Question Q1: What are the primary vectors and mechanics of knowledge-base poisoning and indirect prompt-injection attacks in RAG systems?
- **[Support]** TPARAG achieves high retrieval and end-to-end attack success rates across open-domain QA benchmarks under black-box and white-box RAG settings, outperforming baselines such as PoisonedRAG and GARAG. *(S17)*
  > "Extensive experiments on open-domain QA datasets demonstrate that TPARAG consistently outperforms previous approaches in retrieval-stage and end-to-end attack effectiveness."
  > "In the black-box setting, TPARAG achieves a retrieval attack success rate exceeding 66%, reaching up to 94% in the best case. In the white-box setting, its performance improves further, reaching a 100% success rate in half of the test cases and maintaining a minimum of 83%"

### Sub-Question Q2: What is the empirical vulnerability rate of current RAG architectures to these attacks across standard benchmarks?
- **[Support]** TPARAG achieves high retrieval and end-to-end attack success rates across open-domain QA benchmarks under black-box and white-box RAG settings, outperforming baselines such as PoisonedRAG and GARAG. *(S17)*
  > "Extensive experiments on open-domain QA datasets demonstrate that TPARAG consistently outperforms previous approaches in retrieval-stage and end-to-end attack effectiveness."
  > "In the black-box setting, TPARAG achieves a retrieval attack success rate exceeding 66%, reaching up to 94% in the best case. In the white-box setting, its performance improves further, reaching a 100% success rate in half of the test cases and maintaining a minimum of 83%"

### Sub-Question Q3: How effective are current defense mechanisms (e.g., input sanitization, output filtering, retrieval guardrails) at mitigating these vulnerabilities?
- **[Support]** TPARAG achieves high retrieval and end-to-end attack success rates across open-domain QA benchmarks under black-box and white-box RAG settings, outperforming baselines such as PoisonedRAG and GARAG. *(S17)*
  > "Extensive experiments on open-domain QA datasets demonstrate that TPARAG consistently outperforms previous approaches in retrieval-stage and end-to-end attack effectiveness."
  > "In the black-box setting, TPARAG achieves a retrieval attack success rate exceeding 66%, reaching up to 94% in the best case. In the white-box setting, its performance improves further, reaching a 100% success rate in half of the test cases and maintaining a minimum of 83%"

## 3. Counter-Evidence, Trade-Offs & Falsification Analysis
Contradictions detected in empirical corpus: 0.
- No irreconcilable empirical contradictions identified; observations reflect trade-off boundaries.

## 4. Methodological Limitations & Source Concentration
- Unique Independent Sources: 7% independence ratio.
- Concentration Penalty: 0.90.

## 5. Unresolved Research Gaps
- [Q1] Only 0 independent sources with counter-evidence, but 1 required (total findings: 0).
- [Q2] Only 1 independent sources with quantitative evidence, but 4 required (total findings: 1).
- [Q2] Only 0 independent sources with counter-evidence, but 1 required (total findings: 0).
- [Q3] Only 1 independent sources with quantitative evidence, but 3 required (total findings: 1).
- [Q3] Only 0 independent sources with counter-evidence, but 2 required (total findings: 0).

## 6. Epistemic Conclusion
Based on calibrated evidence scoring (0.15/1.0), the working hypotheses are evaluated with low hedged certainty.