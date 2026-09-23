import json

OUTPUT_FILE = "data/research_evidence.json"


def build_research_evidence_output(
    topic,
    research_package,
    extraction_bundle,
    summary=None,
    sufficiency_results=None,
    research_plan=None
):
    """
    Assemble the complete Phase 2 research_evidence.json artifact.
    Includes structured findings, evidence, sources, gaps, and overall summary.
    """

    canonical_sources = getattr(research_package, "selection_bundle", {}).get("canonical_sources", list(research_package))
    unique_selected = getattr(research_package, "unique_selected_sources", None)
    if unique_selected is None:
        unique_selected = list(research_package)
    retrieval_stats = getattr(research_package, "retrieval_stats", {})

    extracted_questions = extraction_bundle.get("questions", [])
    source_finding_map = extraction_bundle.get("source_finding_map", {})
    all_findings = extraction_bundle.get("all_findings", [])

    # Clean source records for output (omit massive raw text dumps, include metadata needed by Phase 3)
    sources_output = []
    for source in unique_selected:
        source_rec = {
            "source_id": source.get("source_id"),
            "title": source.get("title", ""),
            "url": source.get("url", ""),
            "source_type": source.get("source_type", "other"),
            "content_status": source.get("content_status", "snippet_only"),
            "selected_for_questions": source.get("selected_for_questions", []),
            "selection_reason": source.get("selection_reason", {}),
            "ranking_score": source.get("ranking_score", 0.0),
            "ranking_by_question": source.get("ranking_by_question", {})
        }
        for field in (
            "authors", "publication_year", "doi", "venue", "citation_count",
            "influential_citation_count", "is_open_access", "open_access_url",
            "tldr", "fields_of_study", "provider_name"
        ):
            val = source.get(field)
            if val is not None and val != "" and val != []:
                source_rec[field] = val
        sources_output.append(source_rec)

    # Process evidence gaps and sufficiency status
    all_gaps = []
    sufficiency_evaluation = []
    if sufficiency_results:
        for sr in sufficiency_results:
            qid = sr.get("question_id", "")
            gaps = sr.get("evidence_gaps", [])
            for gap in gaps:
                all_gaps.append(f"[{qid}] {gap}")
            sufficiency_evaluation.append({
                "question_id": qid,
                "sufficient": bool(sr.get("sufficient", False)),
                "selected_source_count": sr.get("selected_source_count", 0),
                "primary_source_count": sr.get("primary_source_count", 0),
                "finding_count": sr.get("finding_count", 0),
                "quantitative_count": sr.get("quantitative_count", 0),
                "counter_count": sr.get("counter_count", 0),
                "missing_requirements": sr.get("missing_requirements", []),
                "evidence_gaps": gaps
            })

    # Process empirical contradictions, evidence diversity, and source independence
    from .summary_generator import detect_empirical_contradictions, calculate_evidence_diversity
    from .evidence_validator import EvidenceValidator
    empirical_contradictions = detect_empirical_contradictions(all_findings)
    diversity_metrics = calculate_evidence_diversity(unique_selected, all_findings)
    independence_metrics = EvidenceValidator.enforce_source_independence(all_findings)

    useful_sources, idle_sources = EvidenceValidator.filter_useful_sources(unique_selected, all_findings)
    useful_ids = {str(s.get("source_id", "")).strip().upper() for s in useful_sources}

    # Clean source records for output (mark useful vs idle)
    sources_output = []
    for source in unique_selected:
        sid = str(source.get("source_id", "")).strip().upper()
        source_rec = {
            "source_id": source.get("source_id"),
            "title": source.get("title", ""),
            "url": source.get("url", ""),
            "source_type": source.get("source_type", "other"),
            "content_status": source.get("content_status", "snippet_only"),
            "is_useful": sid in useful_ids,
            "findings_yielded": len(source_finding_map.get(source.get("source_id"), [])),
            "selected_for_questions": source.get("selected_for_questions", []),
            "selection_reason": source.get("selection_reason", {}),
            "ranking_score": source.get("ranking_score", 0.0),
            "ranking_by_question": source.get("ranking_by_question", {})
        }
        for field in (
            "authors", "publication_year", "doi", "venue", "citation_count",
            "influential_citation_count", "is_open_access", "open_access_url",
            "tldr", "fields_of_study", "provider_name"
        ):
            val = source.get(field)
            if val is not None and val != "" and val != []:
                source_rec[field] = val
        sources_output.append(source_rec)

    # Process evidence gaps and sufficiency status
    all_gaps = []
    sufficiency_evaluation = []
    if sufficiency_results:
        for sr in sufficiency_results:
            qid = sr.get("question_id", "")
            gaps = sr.get("evidence_gaps", [])
            for gap in gaps:
                all_gaps.append(f"[{qid}] {gap}")
            sufficiency_evaluation.append({
                "question_id": qid,
                "sufficient": bool(sr.get("sufficient", False)),
                "selected_source_count": sr.get("selected_source_count", 0),
                "useful_source_count": sr.get("useful_source_count", 0),
                "idle_source_count": sr.get("idle_source_count", 0),
                "primary_source_count": sr.get("primary_source_count", 0),
                "finding_count": sr.get("finding_count", 0),
                "quantitative_count": sr.get("quantitative_count", 0),
                "counter_count": sr.get("counter_count", 0),
                "limitation_count": sr.get("limitation_count", 0),
                "trade_off_count": sr.get("trade_off_count", 0),
                "missing_requirements": sr.get("missing_requirements", []),
                "evidence_gaps": gaps
            })

    # Statistics compilation
    total_findings = len(all_findings)
    support_count = sum(1 for f in all_findings if f.get("stance") == "support")
    counter_count = sum(1 for f in all_findings if f.get("stance") == "counter")
    limitation_count = sum(1 for f in all_findings if f.get("stance") == "limitation")
    trade_off_count = sum(1 for f in all_findings if f.get("stance") in ("trade_off", "mixed"))
    context_count = sum(1 for f in all_findings if f.get("stance") == "context")
    quant_count = sum(1 for f in all_findings if len(f.get("quantitative_evidence", [])) > 0)
    sources_with_findings = len([sid for sid, fids in source_finding_map.items() if len(fids) > 0])

    source_yield_rate = round(len(useful_sources) / max(1, len(unique_selected)), 2)

    statistics = {
        "candidate_sources": len(canonical_sources) + getattr(research_package, "duplicates_merged", 0),
        "canonical_sources": len(canonical_sources),
        "selected_sources": len(unique_selected),
        "useful_sources": len(useful_sources),
        "idle_sources": len(idle_sources),
        "source_yield_rate": source_yield_rate,
        "epistemic_triplet": {
            "finding_count": total_findings,
            "source_count": len(useful_sources),
            "independent_source_count": independence_metrics.get("unique_source_count", 0)
        },
        "retrieval_successes": retrieval_stats.get("successes", 0),
        "retrieval_failures": retrieval_stats.get("failures", 0),
        "sources_with_findings": sources_with_findings,
        "total_findings": total_findings,
        "support_findings": support_count,
        "counter_findings": counter_count,
        "limitation_findings": limitation_count,
        "trade_off_findings": trade_off_count,
        "context_findings": context_count,
        "quantitative_findings": quant_count,
        "unique_independent_sources": independence_metrics.get("unique_source_count", 0),
        "concentration_ratio": independence_metrics.get("concentration_ratio", 0.0),
        "contradictions_detected": len(empirical_contradictions),
        "unmet_evidence_gaps": len(all_gaps),
        "evidence_diversity_score": diversity_metrics.get("diversity_score", 0.0)
    }

    contract_enforcement = None
    if research_plan:
        try:
            from planner.validation.contract_enforcer import enforce_research_contract
            temp_output = {"questions": extracted_questions, "sources": sources_output}
            contract_enforcement = enforce_research_contract(research_plan, temp_output)
        except Exception:
            contract_enforcement = None

    return {
        "schema_version": "1.1",
        "topic": topic,
        "research_plan_reference": "data/research_plan.json",
        "phase": "phase_2_researcher",
        "summary": summary or "",
        "evidence_gaps": all_gaps,
        "sufficiency_evaluation": sufficiency_evaluation,
        "contradictions": empirical_contradictions,
        "evidence_diversity": diversity_metrics,
        "contract_enforcement": contract_enforcement,
        "questions": extracted_questions,
        "sources": sources_output,
        "source_finding_map": source_finding_map,
        "statistics": statistics
    }


def save_research_evidence(
    evidence_data,
    filename=OUTPUT_FILE
):
    """
    Serialize research evidence artifact to research_evidence.json.
    """

    import os
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(
            evidence_data,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nPhase 2 research evidence saved to {filename}")
