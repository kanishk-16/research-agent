MAX_SELECTED_SOURCES_PER_QUESTION = 5
MIN_COUNTER_SELECTION_SCORE = 0.35


def select_sources_for_question(
    question,
    sources,
    max_sources=MAX_SELECTED_SOURCES_PER_QUESTION,
    min_counter_score=MIN_COUNTER_SELECTION_SCORE
):
    """
    Select top ranked sources for one research question, enforcing counter-evidence
    protection when required by the Planner question.
    """

    question_id = str(
        question.get("id", "")
    ).strip().upper()

    requires_counter = bool(
        question.get("requires_counter_evidence", False)
    )

    # 1. Filter sources scored for this question
    eligible_sources = [
        source
        for source in sources
        if question_id in source.get("ranking_by_question", {})
    ]

    # 2. Sort candidates by final score for this specific question
    eligible_sources.sort(
        key=lambda s: s["ranking_by_question"][question_id]["final_score"],
        reverse=True
    )

    if not eligible_sources:
        return {
            "question_id": question_id,
            "selected_sources": [],
            "counter_evidence_available": False,
            "counter_evidence_selected": False,
            "counter_source_id": None
        }

    # 3. Identify counter-evidence candidates for THIS question from discovered_by
    counter_candidates = []
    for source in eligible_sources:
        for discovery in source.get("discovered_by", []):
            disc_qid = str(discovery.get("question_id", "")).strip().upper()
            disc_type = str(discovery.get("query_type", "")).strip().lower()
            if disc_qid == question_id and disc_type == "counter":
                score = source["ranking_by_question"][question_id]["final_score"]
                counter_candidates.append((score, source))
                break

    counter_candidates.sort(key=lambda item: item[0], reverse=True)
    counter_available = len(counter_candidates) > 0

    selected = []
    counter_selected_source = None
    selected_ids = set()

    # 4. If counter evidence is required and a qualified counter source exists, select top counter source
    if requires_counter and counter_available:
        best_counter_score, best_counter_source = counter_candidates[0]
        if best_counter_score >= min_counter_score:
            selected.append((best_counter_source, "counter_evidence"))
            selected_ids.add(best_counter_source["source_id"])
            counter_selected_source = best_counter_source["source_id"]

    # 5. Fill remaining budget with top normal/general candidates for this question
    remaining_budget = max_sources - len(selected)

    for source in eligible_sources:
        if len(selected) >= max_sources:
            break
        if source["source_id"] in selected_ids:
            continue
        selected.append((source, "top_ranked"))
        selected_ids.add(source["source_id"])

    return {
        "question_id": question_id,
        "selected_sources": selected,
        "counter_evidence_available": counter_available,
        "counter_evidence_selected": (counter_selected_source is not None),
        "counter_source_id": counter_selected_source
    }


def select_sources(
    research_questions,
    canonical_sources,
    max_sources_per_question=MAX_SELECTED_SOURCES_PER_QUESTION,
    min_counter_score=MIN_COUNTER_SELECTION_SCORE
):
    """
    Select sources across all research questions, aggregate into unique selected sources,
    and attach question selection provenance.
    """

    per_question_results = {}
    selected_sources_by_id = {}

    for question in research_questions:
        q_res = select_sources_for_question(
            question,
            canonical_sources,
            max_sources=max_sources_per_question,
            min_counter_score=min_counter_score
        )
        question_id = q_res["question_id"]
        per_question_results[question_id] = q_res

        for source, reason in q_res["selected_sources"]:
            source_id = source["source_id"]

            if source_id not in selected_sources_by_id:
                # Store reference to canonical source
                selected_sources_by_id[source_id] = source

                if "selected_for_questions" not in source:
                    source["selected_for_questions"] = []
                if "selection_reason" not in source:
                    source["selection_reason"] = {}

            if question_id not in source["selected_for_questions"]:
                source["selected_for_questions"].append(question_id)

            source["selection_reason"][question_id] = reason

    unique_selected_sources = list(selected_sources_by_id.values())

    # Sort unique selected sources by best ranking score across selected questions or run source ID
    unique_selected_sources.sort(
        key=lambda s: (
            -s.get("ranking_score", 0.0),
            int(s["source_id"][1:]) if s.get("source_id", "").startswith("S") and s["source_id"][1:].isdigit() else 0
        )
    )

    return {
        "unique_selected_sources": unique_selected_sources,
        "per_question_selection": per_question_results,
        "canonical_sources": canonical_sources
    }
