"""
search_agent.summary_generator

Phase 2 - Step 11: Research Summary Generation
Generates a grounded, evidence-backed research summary from the retrieved
sources and extracted findings using Gemini.
"""

from typing import Any, Dict, List, Optional


DEFAULT_MODEL = "gemini-3.5-flash-lite"


def detect_empirical_contradictions(all_findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect empirical tensions or contradictions across findings along specific dimensions:
    model, benchmark/dataset, task, and metric.
    """
    contradictions = []
    findings_by_qid: Dict[str, List[Dict[str, Any]]] = {}
    for f in all_findings:
        qid = f.get("question_id", "GENERAL")
        findings_by_qid.setdefault(qid, []).append(f)

    for qid, f_list in findings_by_qid.items():
        supports = [f for f in f_list if f.get("stance") == "support"]
        counters = [f for f in f_list if f.get("stance") in ("counter", "mixed")]

        if supports and counters:
            supp_models = set()
            supp_datasets = set()
            supp_tasks = set()
            count_models = set()
            count_datasets = set()
            count_tasks = set()

            for s in supports:
                for q in s.get("quantitative_evidence", []):
                    if q.get("model"): supp_models.add(str(q["model"]))
                    if q.get("dataset"): supp_datasets.add(str(q["dataset"]))
                    if q.get("task"): supp_tasks.add(str(q["task"]))

            for c in counters:
                for q in c.get("quantitative_evidence", []):
                    if q.get("model"): count_models.add(str(q["model"]))
                    if q.get("dataset"): count_datasets.add(str(q["dataset"]))
                    if q.get("task"): count_tasks.add(str(q["task"]))

            dim_conflicts = []
            if supp_datasets and count_datasets:
                dim_conflicts.append(f"benchmarks (support: {', '.join(sorted(supp_datasets)[:3])} vs counter: {', '.join(sorted(count_datasets)[:3])})")
            if supp_models and count_models:
                dim_conflicts.append(f"models (support: {', '.join(sorted(supp_models)[:3])} vs counter: {', '.join(sorted(count_models)[:3])})")
            if supp_tasks and count_tasks:
                dim_conflicts.append(f"tasks (support: {', '.join(sorted(supp_tasks)[:3])} vs counter: {', '.join(sorted(count_tasks)[:3])})")

            dim_desc = f" along {'; '.join(dim_conflicts)}" if dim_conflicts else ""

            contradictions.append({
                "question_id": qid,
                "support_claims": [s.get("claim", "") for s in supports[:3]],
                "counter_claims": [c.get("claim", "") for c in counters[:3]],
                "dimensions": {
                    "support_datasets": sorted(supp_datasets),
                    "counter_datasets": sorted(count_datasets),
                    "support_models": sorted(supp_models),
                    "counter_models": sorted(count_models),
                    "support_tasks": sorted(supp_tasks),
                    "counter_tasks": sorted(count_tasks),
                },
                "description": (
                    f"Question {qid} exhibits empirical tension between {len(supports)} supporting finding(s) "
                    f"and {len(counters)} counter/mixed finding(s){dim_desc}."
                )
            })

    return contradictions


def calculate_evidence_diversity(sources: List[Dict[str, Any]], all_findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute evidence diversity metrics across venues, models, benchmarks, and publication years.
    """
    venues = set()
    years = set()
    for s in sources:
        v = s.get("venue")
        if v:
            venues.add(str(v).strip())
        y = s.get("publication_year")
        if y:
            try:
                years.add(int(y))
            except (ValueError, TypeError):
                pass

    models = set()
    benchmarks = set()
    for f in all_findings:
        for q in f.get("quantitative_evidence", []):
            if q.get("model"):
                models.add(str(q["model"]).strip())
            if q.get("dataset"):
                benchmarks.add(str(q["dataset"]).strip())

    venue_score = min(1.0, len(venues) / 3.0) if venues else 0.3
    year_score = min(1.0, len(years) / 2.0) if years else 0.3
    model_score = min(1.0, len(models) / 2.0) if models else 0.3
    benchmark_score = min(1.0, len(benchmarks) / 2.0) if benchmarks else 0.3

    diversity_score = round(
        0.30 * venue_score + 0.25 * benchmark_score + 0.25 * model_score + 0.20 * year_score,
        2
    )

    return {
        "diversity_score": diversity_score,
        "unique_venues": sorted(venues),
        "unique_publication_years": sorted(years),
        "unique_models": sorted(models),
        "unique_benchmarks": sorted(benchmarks)
    }


def build_research_context(
    sources: List[Dict[str, Any]],
    extraction_bundle: Optional[Dict[str, Any]] = None,
    sufficiency_results: Optional[List[Dict[str, Any]]] = None,
    contradictions: Optional[List[Dict[str, Any]]] = None,
    max_sources: int = 12
) -> str:
    """
    Assemble retrieved source text, extracted findings, evidence sufficiency gaps,
    and empirical contradictions into a comprehensive grounded prompt context.
    """
    context_blocks = []

    # 1. Sources context
    for index, source in enumerate(sources[:max_sources], start=1):
        source_id = source.get("source_id", f"S{index}")
        title = source.get("title", "Unknown Title")
        url = source.get("url", "")
        source_type = source.get("source_type", "other")
        content = source.get("content", "").strip()

        # Bibliographic metadata if available
        meta_items = []
        if source.get("authors"):
            auth_list = source["authors"]
            auth_str = ", ".join(auth_list[:2]) + (" et al." if len(auth_list) > 2 else "")
            meta_items.append(f"Authors: {auth_str}")
        if source.get("publication_year"):
            meta_items.append(f"Year: {source['publication_year']}")
        if source.get("venue"):
            meta_items.append(f"Venue: {source['venue']}")
        if source.get("citation_count") is not None:
            meta_items.append(f"Citations: {source['citation_count']}")

        meta_header = f" [{', '.join(meta_items)}]" if meta_items else ""

        # Truncate content snippet to avoid exceeding context budget
        snippet = content[:1500] if content else "(No text content retrieved)"

        context_blocks.append(
            f"--- SOURCE [{source_id}] ({source_type}){meta_header} ---\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content:\n{snippet}\n"
        )

    # 2. Extracted findings summary
    findings_context = ""
    all_findings = []
    if extraction_bundle and isinstance(extraction_bundle, dict):
        all_findings = extraction_bundle.get("all_findings", [])
        if all_findings:
            f_lines = []
            for f in all_findings[:20]:
                fid = f.get("finding_id", "F")
                claim = f.get("claim", "")
                stance = f.get("stance", "context")
                srcs = ", ".join(f.get("source_ids", []))
                f_lines.append(f"- [{fid}] ({stance.upper()}, sources: {srcs}): {claim}")
            findings_context = "\nKEY EXTRACTED FINDINGS:\n" + "\n".join(f_lines) + "\n\n"

    # 3. Sufficiency and Evidence Gaps
    gaps_context = ""
    if sufficiency_results:
        gap_lines = []
        for sr in sufficiency_results:
            qid = sr.get("question_id", "")
            status = "SUFFICIENT" if sr.get("sufficient") else "INSUFFICIENT"
            gaps = sr.get("evidence_gaps", [])
            gap_desc = f" (GAPS: {'; '.join(gaps)})" if gaps else ""
            gap_lines.append(f"- {qid}: {status}{gap_desc}")
        if gap_lines:
            gaps_context = "EVIDENCE SUFFICIENCY & IDENTIFIED GAPS:\n" + "\n".join(gap_lines) + "\n\n"

    # 4. Empirical Contradictions
    contradiction_context = ""
    active_contradictions = contradictions if contradictions is not None else detect_empirical_contradictions(all_findings)
    if active_contradictions:
        c_lines = []
        for c in active_contradictions:
            qid = c.get("question_id", "")
            supp = "; ".join(c.get("support_claims", [])[:2])
            count = "; ".join(c.get("counter_claims", [])[:2])
            c_lines.append(f"- [{qid}] SUPPORT: {supp} | COUNTER: {count}")
        contradiction_context = "IDENTIFIED EMPIRICAL CONTRADICTIONS & TENSIONS:\n" + "\n".join(c_lines) + "\n\n"

    return findings_context + gaps_context + contradiction_context + "\n".join(context_blocks)


def generate_research_summary(
    gemini_client,
    topic: str,
    sources: List[Dict[str, Any]],
    extraction_bundle: Optional[Dict[str, Any]] = None,
    sufficiency_results: Optional[List[Dict[str, Any]]] = None,
    contradictions: Optional[List[Dict[str, Any]]] = None,
    model: str = DEFAULT_MODEL
) -> str:
    """
    Generate a synthesis summary grounded strictly in the retrieved evidence,
    actively addressing evidence gaps and empirical contradictions.
    """
    if not sources:
        return f"No sources retrieved for topic: {topic}"

    context = build_research_context(
        sources,
        extraction_bundle=extraction_bundle,
        sufficiency_results=sufficiency_results,
        contradictions=contradictions
    )

    prompt = f"""You are a scientific research synthesis assistant.

Research Topic:
{topic}

Below are the primary sources, evidence findings, sufficiency status, and identified contradictions gathered by the autonomous researcher:

{context}

Write a comprehensive, evidence-backed research summary addressing the topic.

Requirements:
1. Ground every substantive statement strictly in the provided extracted findings and sources. Do NOT invent facts or numbers.
2. Cite all claims using exact source IDs: [S1], [OA-W...], [S2-...], etc.
3. Explicitly state the empirical consensus and analyze counter-evidence, failure modes, or bottlenecks.
4. Highlight quantitative findings, metrics, and benchmark results when present in the sources.
5. CONTRADICTION ANALYSIS: If empirical contradictions or tensions are identified, explicitly analyze why the findings diverge (e.g. task complexity, baseline sophistication, model scale, coordination protocols).
6. SUMMARY GROUNDING & UNRESOLVED GAPS:
   - Do NOT draw conclusions stronger than the evidence collected.
   - If any research question is marked INSUFFICIENT or lists open evidence gaps, you MUST qualify your conclusions.
   - You MUST include a dedicated concluding section titled '**Unresolved Evidence Gaps & Empirical Uncertainties**' that explicitly details the unresolved gaps and empirical limitations.
7. Write in professional, objective, academic prose (3-5 paragraphs).
"""

    response = gemini_client.models.generate_content(
        model=model,
        contents=prompt
    )

    if not response or not response.text:
        raise RuntimeError("Gemini returned an empty research summary.")

    return response.text.strip()

