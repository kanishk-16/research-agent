"""
Calibrated Evidence Synthesizer (Phase 3).

Synthesizes validated research evidence into a rigorous, epistemic research report.
Dynamically calibrates assertion confidence, hedging language, and uncertainty
based on mathematical sufficiency, source independence, contradiction presence,
and empirical gaps.
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple


def calculate_epistemic_confidence(
    evidence_data: Dict[str, Any],
    research_plan: Optional[Dict[str, Any]] = None
) -> Tuple[float, str, Dict[str, Any]]:
    """
    Computes mathematical Epistemic Confidence Score (0.0 to 1.0) and hedging tier.
    """
    stats = evidence_data.get("statistics", {}) or {}
    sufficiency_list = evidence_data.get("sufficiency_evaluation", []) or []
    contradictions = evidence_data.get("contradictions", []) or []
    diversity_data = evidence_data.get("evidence_diversity", {}) or {}

    total_q = len(sufficiency_list)
    suff_q = sum(1 for s in sufficiency_list if s.get("sufficient", False))
    sufficiency_ratio = (suff_q / total_q) if total_q > 0 else 0.5

    unique_sources = stats.get("unique_independent_sources", 1)
    selected_sources = stats.get("selected_sources", 1)
    independence_ratio = min(1.0, unique_sources / max(1, selected_sources))

    concentration = stats.get("concentration_ratio", 0.0)
    concentration_penalty = max(0.0, concentration - 0.40) * 1.5

    diversity_score = diversity_data.get("diversity_score", 0.5)

    contradiction_count = len(contradictions)
    # Balanced discovery of contradictions is epistemically healthy
    contradiction_modifier = 1.0 if contradiction_count > 0 else 0.90

    raw_score = (
        0.40 * sufficiency_ratio +
        0.30 * independence_ratio +
        0.20 * diversity_score -
        concentration_penalty
    ) * contradiction_modifier

    confidence_score = round(max(0.15, min(0.96, raw_score)), 2)

    if confidence_score >= 0.80:
        tier = "HIGH"
        tier_description = (
            "High Epistemic Confidence: Robust empirical support replicated across multiple "
            "independent primary studies with comprehensive benchmark grounding."
        )
    elif confidence_score >= 0.50:
        tier = "MODERATE"
        tier_description = (
            "Moderate Epistemic Confidence: Observed trends are verified, but constrained by "
            "specific engineering trade-offs, methodological bounds, or limited independent replication."
        )
    else:
        tier = "LOW_HEDGED"
        tier_description = (
            "Preliminary / Qualified Confidence: Evidence exhibits high source concentration, "
            "significant unmet requirements, or unresolved empirical contradictions."
        )

    breakdown = {
        "confidence_score": confidence_score,
        "confidence_tier": tier,
        "tier_description": tier_description,
        "sufficiency_ratio": round(sufficiency_ratio, 2),
        "independence_ratio": round(independence_ratio, 2),
        "concentration_penalty": round(concentration_penalty, 2),
        "diversity_score": round(diversity_score, 2),
        "contradictions_detected": contradiction_count,
    }

    return confidence_score, tier, breakdown


class CalibratedSynthesizer:
    """
    Orchestrates Phase 3 report generation with epistemic calibration.
    """

    def __init__(self, gemini_client=None, model="gemini-2.5-flash"):
        self.gemini_client = gemini_client
        self.model = model

    def synthesize(
        self,
        topic: str,
        research_plan: Dict[str, Any],
        evidence_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Produce a calibrated research synthesis artifact.
        """
        confidence_score, tier, breakdown = calculate_epistemic_confidence(evidence_data, research_plan)

        sources = evidence_data.get("sources", [])
        questions = evidence_data.get("questions", [])
        contradictions = evidence_data.get("contradictions", [])
        gaps = evidence_data.get("evidence_gaps", [])
        stats = evidence_data.get("statistics", {})

        # Build prompt for LLM synthesis
        prompt = self._build_synthesis_prompt(
            topic, research_plan, questions, sources,
            contradictions, gaps, confidence_score, tier, breakdown
        )

        report_markdown = None
        if self.gemini_client is not None:
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                if response and response.text:
                    report_markdown = response.text.strip()
            except Exception as exc:
                print(f"[Warning] LLM synthesis call failed: {exc}. Generating deterministic fallback report.")

        if not report_markdown:
            report_markdown = self._generate_deterministic_report(
                topic, research_plan, questions, sources,
                contradictions, gaps, confidence_score, tier, breakdown
            )

        report_artifact = {
            "schema_version": "1.0",
            "phase": "phase_3_synthesizer",
            "topic": topic,
            "epistemic_calibration": breakdown,
            "report_markdown": report_markdown,
            "evidence_reference": "data/research_evidence.json",
            "plan_reference": "data/research_plan.json"
        }

        return report_artifact

    def _build_synthesis_prompt(
        self, topic, plan, questions, sources,
        contradictions, gaps, confidence, tier, breakdown
    ) -> str:
        source_summary = "\n".join(
            f"[{s.get('source_id')}]: \"{s.get('title')}\" ({s.get('venue') or 'Unknown Venue'}, {s.get('publication_year') or 'n.d.'})"
            for s in sources[:12]
        )

        findings_summary = []
        for q in questions:
            qid = q.get("question_id")
            q_text = q.get("question")
            f_list = q.get("findings", [])
            findings_summary.append(f"\n### Sub-Question {qid}: {q_text}")
            for f in f_list:
                sids = ", ".join(f.get("source_ids", []))
                stance = f.get("stance", "context").upper()
                claim = f.get("claim", "")
                findings_summary.append(f"- [{stance}] {claim} (Cited in: {sids})")

        return f"""
You are the Lead Scientific Synthesizer of the Autonomous Research Agent (AARA).
Generate a comprehensive, scientifically rigorous academic report for the following topic:

TOPIC: {topic}

EPISTEMIC CALIBRATION LEVEL:
- Confidence Score: {confidence} / 1.0 ({tier})
- Calibration Directive: {breakdown['tier_description']}

IMPORTANT HEDGING RULES:
- If tier is HIGH: Use assertive, evidence-grounded prose; cite independent sources directly.
- If tier is MODERATE: Use nuanced phrasing; emphasize specific trade-offs (e.g. latency vs accuracy) and domain boundaries.
- If tier is LOW_HEDGED: Strictly qualify claims; highlight empirical gaps, lack of independent consensus, and conflicting observations.
- Always cite sources by their identifier (e.g., [S1], [S2]).
- Never invent numbers or citations outside the verified findings.

AVAILABLE SOURCES:
{source_summary}

VERIFIED EMPIRICAL FINDINGS:
{' '.join(findings_summary)}

IDENTIFIED CONTRADICTIONS & DISSONANCES:
{json.dumps(contradictions, indent=2)}

UNRESOLVED EVIDENCE GAPS:
{json.dumps(gaps, indent=2)}

REPORT STRUCTURE REQUIREMENTS:
# {topic}
## 1. Executive Summary & Epistemic Calibration
## 2. Evidence Analysis by Sub-Question
## 3. Quantitative Evaluation & Comparative Contrasts
## 4. Counter-Evidence, Trade-Offs & Falsification Analysis
## 5. Methodological Limitations & Source Independence
## 6. Unresolved Research Gaps & Epistemic Conclusion
"""

    def _generate_deterministic_report(
        self, topic, plan, questions, sources,
        contradictions, gaps, confidence, tier, breakdown
    ) -> str:
        lines = [
            f"# Research Report: {topic}",
            "",
            "## 1. Executive Summary & Epistemic Calibration",
            f"**Epistemic Confidence Rating**: `{confidence} / 1.0` (**{tier}**)",
            f"> {breakdown['tier_description']}",
            "",
            f"This research synthesis evaluates empirical evidence across {len(questions)} sub-questions, citing {len(sources)} independent sources.",
            "",
            "## 2. Evidence Analysis by Sub-Question"
        ]

        for q in questions:
            qid = q.get("question_id")
            lines.append(f"### Sub-Question {qid}: {q.get('question')}")
            findings = q.get("findings", [])
            if findings:
                for f in findings:
                    sids = ", ".join(f.get("source_ids", []))
                    stance = f.get("stance", "context").capitalize()
                    lines.append(f"- **[{stance}]** {f.get('claim')} *({sids})*")
                    for ev in f.get("evidence", []):
                        lines.append(f"  > \"{ev.get('evidence_text')}\"")
            else:
                lines.append("- *No verified findings extracted for this sub-question.*")
            lines.append("")

        lines.extend([
            "## 3. Counter-Evidence, Trade-Offs & Falsification Analysis",
            f"Contradictions detected in empirical corpus: {len(contradictions)}."
        ])
        if contradictions:
            for c in contradictions:
                lines.append(f"- **Contradiction**: {c.get('description', 'Opposing empirical claims observed.')}")
        else:
            lines.append("- No irreconcilable empirical contradictions identified; observations reflect trade-off boundaries.")
        lines.append("")

        lines.extend([
            "## 4. Methodological Limitations & Source Concentration",
            f"- Unique Independent Sources: {breakdown['independence_ratio'] * 100:.0f}% independence ratio.",
            f"- Concentration Penalty: {breakdown['concentration_penalty']:.2f}.",
            "",
            "## 5. Unresolved Research Gaps",
        ])
        if gaps:
            for g in gaps:
                lines.append(f"- {g}")
        else:
            lines.append("- All planned primary and quantitative evidence requirements satisfied.")
        lines.append("")

        lines.extend([
            "## 6. Epistemic Conclusion",
            f"Based on calibrated evidence scoring ({confidence}/1.0), the working hypotheses are evaluated with {tier.lower().replace('_', ' ')} certainty."
        ])

        return "\n".join(lines)


def save_research_report(report_data: Dict[str, Any], md_path="data/research_report.md", json_path="data/research_report.json"):
    """
    Saves synthesized report artifacts.
    """
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_data.get("report_markdown", ""))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 3 research report saved to {md_path} and {json_path}")

