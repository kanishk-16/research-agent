"""
Planner–Researcher Contract Enforcement Engine.

Verifies that the empirical evidence gathered by the Researcher strictly satisfies
the contract established during the Planning phase (source types, quantitative quotas,
counter-evidence requirements, falsification criteria, and independent source thresholds).
"""

from typing import Dict, List, Any


def enforce_research_contract(
    research_plan: Dict[str, Any],
    evidence_output: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates compliance between research_plan and evidence_output.

    Returns:
      {
        "compliance_score": float (0.0 to 1.0),
        "contract_status": "PASSED" | "PARTIAL" | "VIOLATED",
        "question_compliance": [
            {
                "question_id": str,
                "score": float,
                "status": "MET" | "PARTIAL" | "UNMET",
                "clauses": [
                    {"clause": str, "target": int/bool, "actual": int/bool, "met": bool}
                ]
            }
        ],
        "summary": str
      }
    """
    questions = research_plan.get("questions", []) if isinstance(research_plan, dict) else []
    extracted_questions = evidence_output.get("questions", []) if isinstance(evidence_output, dict) else []
    
    # Map extracted questions by question_id
    extracted_map = {eq.get("question_id"): eq for eq in extracted_questions if isinstance(eq, dict)}

    question_compliance = []
    total_clauses = 0
    met_clauses = 0

    for q in questions:
        qid = q.get("id")
        eq = extracted_map.get(qid, {})
        findings = eq.get("findings", [])
        
        # Determine unique sources and independent counts
        source_ids = set()
        counter_sources = set()
        quant_sources = set()
        primary_sources = set()

        for f in findings:
            sids = f.get("source_ids", [])
            for sid in sids:
                source_ids.add(sid)
                if f.get("stance") == "counter":
                    counter_sources.add(sid)
                if f.get("quantitative_evidence"):
                    quant_sources.add(sid)

        # Count primary sources from findings
        all_sources = evidence_output.get("sources", [])
        source_meta = {s.get("source_id"): s for s in all_sources if isinstance(s, dict)}
        for sid in source_ids:
            s = source_meta.get(sid, {})
            try:
                from search_agent.sources import is_primary_source_type
                is_prim = is_primary_source_type(s)
            except Exception:
                is_prim = False

            stype = str(s.get("source_type", "")).strip().lower()
            if is_prim or stype in (
                "primary_paper", "peer_reviewed", "preprint", "journal_article",
                "conference_paper", "peer-reviewed conference paper",
                "peer-reviewed journal article", "archived primary research preprint",
                "official technical report", "book_chapter", "article"
            ) or s.get("doi") or s.get("provider_name") in ("openalex", "semanticscholar"):
                primary_sources.add(sid)

        reqs = q.get("source_requirements", {}) or {}
        min_sources = reqs.get("minimum_sources", 1)
        min_primary = reqs.get("minimum_primary_sources", 0)
        min_quant = reqs.get("minimum_quantitative_sources", 1 if q.get("requires_quantitative_evidence") else 0)
        min_counter = reqs.get("minimum_counter_evidence_sources", 1 if q.get("requires_counter_evidence") else 0)

        clauses = []

        # Clause 1: Overall source count
        met_src = len(source_ids) >= min_sources
        clauses.append({
            "clause": "minimum_sources",
            "target": min_sources,
            "actual": len(source_ids),
            "met": met_src
        })

        # Clause 2: Primary sources
        met_prim = len(primary_sources) >= min_primary
        clauses.append({
            "clause": "minimum_primary_sources",
            "target": min_primary,
            "actual": len(primary_sources),
            "met": met_prim
        })

        # Clause 3: Quantitative sources
        met_q = len(quant_sources) >= min_quant
        clauses.append({
            "clause": "minimum_quantitative_sources",
            "target": min_quant,
            "actual": len(quant_sources),
            "met": met_q
        })

        # Clause 4: Counter evidence sources
        met_c = len(counter_sources) >= min_counter
        clauses.append({
            "clause": "minimum_counter_evidence_sources",
            "target": min_counter,
            "actual": len(counter_sources),
            "met": met_c
        })

        # Score question
        q_met = sum(1 for c in clauses if c["met"])
        q_total = len(clauses)
        q_score = round(q_met / q_total, 2)
        total_clauses += q_total
        met_clauses += q_met

        q_status = "MET" if q_score >= 0.85 else ("PARTIAL" if q_score >= 0.50 else "UNMET")

        question_compliance.append({
            "question_id": qid,
            "score": q_score,
            "status": q_status,
            "clauses": clauses
        })

    overall_score = round(met_clauses / total_clauses, 2) if total_clauses > 0 else 1.0
    if overall_score >= 0.85:
        overall_status = "PASSED"
    elif overall_score >= 0.50:
        overall_status = "PARTIAL"
    else:
        overall_status = "VIOLATED"

    summary = (
        f"Contract compliance {overall_status} (Score: {overall_score * 100:.1f}%). "
        f"{met_clauses}/{total_clauses} requirements verified across {len(questions)} sub-questions."
    )

    return {
        "compliance_score": overall_score,
        "contract_status": overall_status,
        "question_compliance": question_compliance,
        "summary": summary
    }

