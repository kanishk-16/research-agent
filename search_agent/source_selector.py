import re
from .sources import is_primary_source_type, PRIMARY_SOURCE_TYPES

MAX_SELECTED_SOURCES_PER_QUESTION = 5
MIN_COUNTER_SELECTION_SCORE = 0.35
MAX_SOURCES_PER_VENUE = 2
MAX_SOURCES_PER_AUTHOR = 2

DISCOVERY_ONLY_SOURCE_TYPES = {
    "corporate_blog",
    "personal_blog",
    "forum",
    "social_media",
    "video",
    "news",
}


_CONTENT_STATUS_ORDER = {
    "full": 4,
    "partial": 3,
    "snippet_only": 2,
    "failed": 1,
}


def _extract_source_venue_or_domain(source):
    venue = str(source.get("venue", "") or "").strip().lower()
    if venue:
        return venue
    url = str(source.get("url", "") or "").strip().lower()
    if url:
        from urllib.parse import urlsplit
        hostname = urlsplit(url).hostname or ""
        if hostname:
            return hostname.lower()
    return "unknown"


def _extract_primary_author(source):
    authors = source.get("authors") or []
    if authors and isinstance(authors, list):
        return str(authors[0]).strip().lower()
    return "unknown"


def _is_quantitative_candidate(src, qid):
    if src.get("ranking_by_question", {}).get(qid, {}).get("quantitative_bonus", 0.0) > 0:
        return True
    text = f"{src.get('title', '')} {src.get('abstract', '')} {src.get('content', '')}".lower()
    return bool(re.search(r"\b\d+(?:\.\d+)?%|\b\d+\.\d+\b|\baccuracy\b|\bf1\b|\bpass@1\b|\bbaseline\b|\bscore\b", text))


def select_sources_for_question(
    question,
    sources,
    max_sources=MAX_SELECTED_SOURCES_PER_QUESTION,
    min_counter_score=MIN_COUNTER_SELECTION_SCORE
):
    """
    Select top ranked sources for one research question, actively satisfying Planner requirements
    (minimum primary sources, minimum counter-evidence sources, minimum quantitative sources)
    and enforcing venue/author diversity.
    Strictly excludes discovery-only sources and failed extraction sources.
    """

    question_id = str(
        question.get("id", "")
    ).strip().upper()

    requires_counter = bool(
        question.get("requires_counter_evidence", False)
    )
    requires_quant = bool(
        question.get("requires_quantitative_evidence", False)
    )
    source_reqs = question.get("source_requirements", {}) or {}
    min_counter_req = int(source_reqs.get("minimum_counter_evidence_sources", 1 if requires_counter else 0))
    min_primary_req = int(source_reqs.get("minimum_primary_sources", 0))
    min_quant_req = int(source_reqs.get("minimum_quantitative_sources", 1 if requires_quant else 0))
    min_sources_quota = int(source_reqs.get("minimum_sources", 0))

    # Dynamic cap: ensure effective_max satisfies required quotas
    effective_max = max(max_sources, min_sources_quota, min_primary_req + min_counter_req)

    # 1. Filter sources scored for this question, strictly eligible as evidence, and with non-negligible relevance
    eligible_sources = [
        source
        for source in sources
        if question_id in source.get("ranking_by_question", {})
        and source.get("evidence_eligible", True) is True
        and source.get("eligibility_reason") not in ("policy_discovery_only", "content_extraction_failed")
        and source.get("source_type") not in DISCOVERY_ONLY_SOURCE_TYPES
        and float(source.get("ranking_by_question", {}).get(question_id, {}).get("relevance_score", 0.0) or 0.0) >= 0.15
    ]

    # 2. Sort candidates by final score for this specific question,
    #    using empirical quantitative content and content_status as tiebreakers
    eligible_sources.sort(
        key=lambda s: (
            s["ranking_by_question"][question_id]["final_score"],
            1 if _is_quantitative_candidate(s, question_id) else 0,
            _CONTENT_STATUS_ORDER.get(
                str(s.get("content_status", "") or "").lower(),
                0
            )
        ),
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
    selected_ids = set()
    venue_counts = {}
    author_counts = {}

    def _can_add_source_diversity(src):
        v = _extract_source_venue_or_domain(src)
        a = _extract_primary_author(src)
        venue_ok = (v == "unknown") or (venue_counts.get(v, 0) < MAX_SOURCES_PER_VENUE)
        author_ok = (a == "unknown") or (author_counts.get(a, 0) < MAX_SOURCES_PER_AUTHOR)
        return venue_ok and author_ok

    def _add_source(src, role):
        selected.append((src, role))
        selected_ids.add(src["source_id"])
        v = _extract_source_venue_or_domain(src)
        venue_counts[v] = venue_counts.get(v, 0) + 1
        a = _extract_primary_author(src)
        author_counts[a] = author_counts.get(a, 0) + 1

    # 4. Actively satisfy counter-evidence requirement
    counter_target = min_counter_req if (requires_counter or min_counter_req > 0) else 0
    counter_selected_ids = []
    if counter_target > 0 and counter_available:
        for c_score, c_src in counter_candidates:
            if len(counter_selected_ids) >= counter_target or len(selected) >= effective_max:
                break
            if c_score >= min_counter_score and c_src["source_id"] not in selected_ids:
                if _can_add_source_diversity(c_src):
                    _add_source(c_src, "counter_evidence")
                    counter_selected_ids.append(c_src["source_id"])

        # If diversity cap blocked counter fulfillment, relax diversity for counter evidence
        if len(counter_selected_ids) < counter_target:
            for c_score, c_src in counter_candidates:
                if len(counter_selected_ids) >= counter_target or len(selected) >= effective_max:
                    break
                if c_score >= min_counter_score and c_src["source_id"] not in selected_ids:
                    _add_source(c_src, "counter_evidence")
                    counter_selected_ids.append(c_src["source_id"])

    # 5. Actively satisfy primary source requirement
    primary_candidates = [
        s for s in eligible_sources
        if is_primary_source_type(s) and s["source_id"] not in selected_ids
    ]
    current_primary_count = sum(
        1 for s, _ in selected if is_primary_source_type(s)
    )
    for p_src in primary_candidates:
        if current_primary_count >= min_primary_req or len(selected) >= effective_max:
            break
        if _can_add_source_diversity(p_src):
            _add_source(p_src, "top_ranked")
            current_primary_count += 1

    # 6. Actively satisfy quantitative evidence requirement
    if min_quant_req > 0:
        current_quant_count = sum(
            1 for s, _ in selected if _is_quantitative_candidate(s, question_id)
        )
        quant_candidates = [
            s for s in eligible_sources
            if _is_quantitative_candidate(s, question_id) and s["source_id"] not in selected_ids
        ]
        for q_src in quant_candidates:
            if current_quant_count >= min_quant_req or len(selected) >= effective_max:
                break
            if _can_add_source_diversity(q_src):
                _add_source(q_src, "quantitative_evidence")
                current_quant_count += 1

        if current_quant_count < min_quant_req:
            for q_src in quant_candidates:
                if current_quant_count >= min_quant_req or len(selected) >= effective_max:
                    break
                if q_src["source_id"] not in selected_ids:
                    _add_source(q_src, "quantitative_evidence")
                    current_quant_count += 1

    # 7. Fill remaining budget respecting venue diversity cap
    for source in eligible_sources:
        if len(selected) >= effective_max:
            break
        if source["source_id"] in selected_ids:
            continue
        if _can_add_source_diversity(source):
            _add_source(source, "top_ranked")

    # 8. Only relax diversity cap if minimum required sources quota is unmet
    if len(selected) < min_sources_quota:
        for source in eligible_sources:
            if len(selected) >= min_sources_quota or len(selected) >= effective_max:
                break
            if source["source_id"] in selected_ids:
                continue
            _add_source(source, "top_ranked")


    primary_counter_id = counter_selected_ids[0] if counter_selected_ids else None

    return {
        "question_id": question_id,
        "selected_sources": selected,
        "counter_evidence_available": counter_available,
        "counter_evidence_selected": len(counter_selected_ids) > 0,
        "counter_source_id": primary_counter_id
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
            -_CONTENT_STATUS_ORDER.get(
                str(s.get("content_status", "") or "").lower(),
                0
            ),
            int(s["source_id"][1:]) if s.get("source_id", "").startswith("S") and s["source_id"][1:].isdigit() else 0
        )
    )

    return {
        "unique_selected_sources": unique_selected_sources,
        "per_question_selection": per_question_results,
        "canonical_sources": canonical_sources
    }
