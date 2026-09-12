import re


def _count_primary_sources(selected_sources):
    primary_types = {
        "peer_reviewed_paper",
        "conference_paper",
        "preprint",
        "systematic_review",
    }
    count = 0
    for src, _ in selected_sources:
        st = str(src.get("source_type", "") or "").strip().lower()
        if st in primary_types:
            count += 1
    return count


def _count_peer_reviewed(selected_sources):
    pr_types = {
        "peer_reviewed_paper",
        "conference_paper",
    }
    count = 0
    for src, _ in selected_sources:
        st = str(src.get("source_type", "") or "").strip().lower()
        if st in pr_types:
            count += 1
    return count


def evaluate_evidence_sufficiency(
    question,
    extraction_findings,
    selection_bundle
):
    """
    Evaluate whether a single research question has sufficient evidence
    according to its requirements in the research plan.

    Returns a dict with:
      - question_id
      - sufficient: bool
      - selected_source_count
      - successful_retrieval_count
      - finding_count
      - support_count
      - counter_count
      - quantitative_count
      - primary_source_count
      - peer_reviewed_count
      - missing_requirements: list[str]
      - evidence_gaps: list[str]
    """

    question_id = str(question.get("id", "")).strip().upper()
    source_reqs = question.get("source_requirements", {}) or {}
    requires_quant = bool(question.get("requires_quantitative_evidence", False))
    requires_counter = bool(question.get("requires_counter_evidence", False))

    q_sel_data = selection_bundle.get("per_question_selection", {}).get(question_id, {})
    selected_sources = q_sel_data.get("selected_sources", [])

    selected_source_count = len(selected_sources)
    successful_retrieval_count = sum(
        1 for src, _ in selected_sources
        if str(src.get("content_status", "") or "").lower() in ("full", "partial")
    )

    support_count = sum(1 for f in extraction_findings if f.get("stance") == "support")
    counter_count = sum(1 for f in extraction_findings if f.get("stance") == "counter")
    quantitative_count = sum(
        1 for f in extraction_findings
        if len(f.get("quantitative_evidence", [])) > 0
    )
    finding_count = len(extraction_findings)

    primary_source_count = _count_primary_sources(selected_sources)
    peer_reviewed_count = _count_peer_reviewed(selected_sources)

    min_sources = source_reqs.get("minimum_sources", 0)
    min_primary = source_reqs.get("minimum_primary_sources", 0)
    min_quant = source_reqs.get("minimum_quantitative_sources", 0)
    min_counter = source_reqs.get("minimum_counter_evidence_sources", 0)
    min_peer_reviewed = source_reqs.get("minimum_peer_reviewed", 0)

    missing_requirements = []
    evidence_gaps = []

    if selected_source_count < min_sources:
        missing_requirements.append("insufficient_sources")
        evidence_gaps.append(
            f"Selected {selected_source_count} sources, "
            f"but {min_sources} required."
        )

    if min_primary > 0 and primary_source_count < min_primary:
        missing_requirements.append("insufficient_primary_sources")
        evidence_gaps.append(
            f"Only {primary_source_count} primary sources, "
            f"but {min_primary} required."
        )

    if min_peer_reviewed > 0 and peer_reviewed_count < min_peer_reviewed:
        missing_requirements.append("insufficient_peer_reviewed")
        evidence_gaps.append(
            f"Only {peer_reviewed_count} peer-reviewed sources, "
            f"but {min_peer_reviewed} required."
        )

    if requires_quant and quantitative_count < min_quant:
        missing_requirements.append("quantitative_evidence")
        evidence_gaps.append(
            f"Only {quantitative_count} quantitative findings, "
            f"but {min_quant} required."
        )

    if requires_counter and counter_count < min_counter:
        missing_requirements.append("counter_evidence")
        evidence_gaps.append(
            f"Only {counter_count} counter-evidence findings, "
            f"but {min_counter} required."
        )

    if successful_retrieval_count == 0 and finding_count == 0:
        evidence_gaps.append(
            "No successful retrievals and no findings extracted."
        )

    sufficient = len(missing_requirements) == 0

    return {
        "question_id": question_id,
        "sufficient": sufficient,
        "selected_source_count": selected_source_count,
        "successful_retrieval_count": successful_retrieval_count,
        "finding_count": finding_count,
        "support_count": support_count,
        "counter_count": counter_count,
        "quantitative_count": quantitative_count,
        "primary_source_count": primary_source_count,
        "peer_reviewed_count": peer_reviewed_count,
        "missing_requirements": missing_requirements,
        "evidence_gaps": evidence_gaps,
    }


def generate_targeted_queries(question, missing_requirements):
    """
    Generate deterministic targeted follow-up queries for a question
    based on its missing evidence requirements.

    Does NOT use LLM calls. Uses template-based generation from the
    question text and missing requirement types.
    """

    question_text = str(question.get("question", "") or "").strip()
    qtype = str(question.get("type", "") or "").strip().lower()
    qtext_lower = question_text.lower()

    queries = []

    for req in missing_requirements:
        req = str(req).strip().lower()

        if req == "quantitative_evidence":
            if "rag" in qtext_lower and "hallucination" in qtext_lower:
                queries.append({
                    "query_text": (
                        "RAG hallucination rate quantitative comparison "
                        "benchmark baseline LLM"
                    ),
                    "query_type": "normal",
                })
                queries.append({
                    "query_text": (
                        "RAG vs non-RAG hallucination percentage "
                        "empirical results benchmark"
                    ),
                    "query_type": "normal",
                })
            elif "performance" in qtext_lower or "benchmark" in qtext_lower:
                queries.append({
                    "query_text": (
                        "quantitative performance comparison "
                        "benchmark results evaluation"
                    ),
                    "query_type": "normal",
                })
            else:
                queries.append({
                    "query_text": (
                        f"{question_text} quantitative results empirical data"
                    ),
                    "query_type": "normal",
                })

        elif req == "counter_evidence":
            if "rag" in qtext_lower and "hallucination" in qtext_lower:
                queries.append({
                    "query_text": (
                        "RAG fails to reduce hallucinations failure cases "
                        "retrieval noise increases errors"
                    ),
                    "query_type": "counter",
                })
                queries.append({
                    "query_text": (
                        "retrieval augmented generation hallucination failure "
                        "limitations benchmark study"
                    ),
                    "query_type": "counter",
                })
            elif "performance" in qtext_lower or "comparison" in qtext_lower:
                queries.append({
                    "query_text": (
                        "counter evidence limitations failures "
                        "negative results comparison"
                    ),
                    "query_type": "counter",
                })
            else:
                queries.append({
                    "query_text": (
                        f"{question_text} limitations failures counter evidence"
                    ),
                    "query_type": "counter",
                })

        elif req == "insufficient_sources":
            queries.append({
                "query_text": question_text,
                "query_type": "normal",
            })

        elif req == "insufficient_primary_sources":
            if "rag" in qtext_lower or "hallucination" in qtext_lower:
                queries.append({
                    "query_text": (
                        "peer-reviewed RAG hallucination "
                        "primary research study"
                    ),
                    "query_type": "normal",
                })
            else:
                queries.append({
                    "query_text": (
                        f"{question_text} peer-reviewed primary research"
                    ),
                    "query_type": "normal",
                })

        elif req == "insufficient_peer_reviewed":
            queries.append({
                "query_text": (
                    f"{question_text} peer-reviewed journal conference paper"
                ),
                "query_type": "normal",
            })

    seen = set()
    unique_queries = []
    for q in queries:
        key = (q["query_text"].strip().lower(), q["query_type"])
        if key not in seen:
            seen.add(key)
            unique_queries.append(q)

    return unique_queries


def check_all_sufficiency(questions, extraction_bundle, selection_bundle):
    """
    Evaluate evidence sufficiency for all questions.

    Returns:
      - results: list of sufficiency dicts, one per question
      - insufficient_questions: list of (question, sufficiency_result) tuples
    """

    results = []
    insufficient_questions = []

    questions_by_id = {
        str(q.get("id", "")).strip().upper(): q
        for q in questions
    }

    extracted_questions = extraction_bundle.get("questions", [])

    for eq in extracted_questions:
        qid = str(eq.get("question_id", "")).strip().upper()
        question = questions_by_id.get(qid)
        if question is None:
            continue

        findings = eq.get("findings", [])
        sufficiency = evaluate_evidence_sufficiency(
            question, findings, selection_bundle
        )
        results.append(sufficiency)

        if not sufficiency["sufficient"]:
            insufficient_questions.append((question, sufficiency))

    return results, insufficient_questions
