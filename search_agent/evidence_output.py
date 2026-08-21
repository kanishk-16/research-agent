import json

OUTPUT_FILE = "research_evidence.json"


def build_research_evidence_output(
    topic,
    research_package,
    extraction_bundle
):
    """
    Assemble the complete Phase 2 research_evidence.json artifact.
    """

    canonical_sources = getattr(research_package, "selection_bundle", {}).get("canonical_sources", list(research_package))
    unique_selected = getattr(research_package, "unique_selected_sources", [])
    retrieval_stats = getattr(research_package, "retrieval_stats", {})

    extracted_questions = extraction_bundle.get("questions", [])
    source_finding_map = extraction_bundle.get("source_finding_map", {})
    all_findings = extraction_bundle.get("all_findings", [])

    # Clean source records for output (omit massive raw text dumps, include metadata needed by Phase 3)
    sources_output = []
    for source in unique_selected:
        sources_output.append({
            "source_id": source.get("source_id"),
            "title": source.get("title", ""),
            "url": source.get("url", ""),
            "source_type": source.get("source_type", "other"),
            "content_status": source.get("content_status", "snippet_only"),
            "selected_for_questions": source.get("selected_for_questions", []),
            "selection_reason": source.get("selection_reason", {}),
            "ranking_score": source.get("ranking_score", 0.0),
            "ranking_by_question": source.get("ranking_by_question", {})
        })

    # Statistics compilation
    total_findings = len(all_findings)
    support_count = sum(1 for f in all_findings if f.get("stance") == "support")
    counter_count = sum(1 for f in all_findings if f.get("stance") == "counter")
    mixed_count = sum(1 for f in all_findings if f.get("stance") == "mixed")
    context_count = sum(1 for f in all_findings if f.get("stance") == "context")
    quant_count = sum(1 for f in all_findings if len(f.get("quantitative_evidence", [])) > 0)
    sources_with_findings = len([sid for sid, fids in source_finding_map.items() if len(fids) > 0])

    statistics = {
        "candidate_sources": len(canonical_sources) + getattr(research_package, "duplicates_merged", 0),
        "canonical_sources": len(canonical_sources),
        "selected_sources": len(unique_selected),
        "retrieval_successes": retrieval_stats.get("successes", 0),
        "retrieval_failures": retrieval_stats.get("failures", 0),
        "sources_with_findings": sources_with_findings,
        "total_findings": total_findings,
        "support_findings": support_count,
        "counter_findings": counter_count,
        "mixed_findings": mixed_count,
        "context_findings": context_count,
        "quantitative_findings": quant_count
    }

    return {
        "schema_version": "1.0",
        "topic": topic,
        "research_plan_reference": "research_plan.json",
        "phase": "phase_2_researcher",
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

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(
            evidence_data,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nPhase 2 research evidence saved to {filename}")
