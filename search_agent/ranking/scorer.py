"""
search_agent/ranking/scorer.py

Zero-shot subject vector similarity gating, per-question source scoring,
ranking, and diagnostic reporting.
"""

import math
import re

from ..semantic_indexer import cosine_similarity, stem_word
from .policy import (
    RANKING_WEIGHTS,
    SOURCE_QUALITY_SCORES,
    SOURCE_TYPES,
    _build_policy_quality_scores,
    _planner_preference_score,
    _tavily_score,
    classify_source,
    compute_evidence_eligibility,
)


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "if", "in", "into", "is", "it", "its", "of", "on", "or",
    "that", "the", "their", "these", "this", "to", "under", "what", "when",
    "where", "which", "while", "who", "whom", "why", "with"
}

_METHODOLOGY_BOILERPLATE_TERMS = {
    "quantitative", "quantitatively", "qualitative", "qualitatively",
    "comparison", "comparative", "comparisons", "compare", "compared", "comparing",
    "assessment", "assess", "assessing", "evaluating", "evaluation", "evaluations",
    "evaluate", "evaluated", "analysis", "analyses", "analyzing", "analyze",
    "study", "studies", "empirical", "empirically", "experimental", "experimentally",
    "experiment", "experiments", "counter", "evidence", "approach", "approaches",
    "method", "methods", "methodology", "methodologies", "using", "based", "new",
    "novel", "paper", "result", "results", "investigation", "investigating",
    "systematic", "overview", "survey", "review", "effective", "effectiveness",
    "effectively", "impact", "impacts", "role", "framework", "frameworks",
    "technique", "techniques", "application", "applications", "standardized",
    "standard", "complex", "benchmark", "benchmarks", "benchmarking", "baseline",
    "baselines", "system", "systems", "performance", "performances", "workflow",
    "workflows", "calibration", "trade", "off", "offs", "tradeoff", "tradeoffs",
    "failure", "failures", "mode", "modes", "introduced", "introduce", "applying",
    "apply", "applied", "during", "resolution", "resolving", "resolve", "condition",
    "conditions", "against", "across", "different", "current", "primary", "high",
    "low", "large", "small", "model", "models", "rate", "rates", "score", "scores",
    "metric", "metrics", "measure", "measurement", "measurements", "effect", "effects",
    "challenge", "challenges", "limitation", "limitations", "boundary", "factor",
    "factors", "level", "levels", "non", "time", "data", "task", "tasks"
}

_ACRONYM_EXPANSIONS = {
    "llm": ("large", "language", "model"),
    "llms": ("large", "language", "models"),
    "rag": ("retrieval", "augmented", "generation"),
    "nlp": ("natural", "language", "processing"),
    "cot": ("chain", "thought"),
    "ml": ("machine", "learning"),
    "ai": ("artificial", "intelligence"),
    "sd": ("speculative", "decoding"),
}


def _stem_token(token: str) -> str:
    return stem_word(token)


_METHODOLOGY_BOILERPLATE_STEMS = {_stem_token(w) for w in _METHODOLOGY_BOILERPLATE_TERMS}


def _tokens(value):
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value).lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _extract_subject_tokens_and_stems(text: str, expand_acronyms: bool = True):
    raw_tokens = [
        t for t in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(t) > 1 and t not in STOPWORDS
    ]
    expanded = list(raw_tokens)
    if expand_acronyms:
        token_set = set(raw_tokens)
        for acr, full_words in _ACRONYM_EXPANSIONS.items():
            if acr in token_set:
                expanded.extend(full_words)
            elif all(fw in token_set for fw in full_words):
                expanded.append(acr)

    subject_tokens = []
    subject_stems = []
    for t in expanded:
        if len(t) <= 2 and t not in _ACRONYM_EXPANSIONS:
            continue
        st = _stem_token(t)
        if t in _METHODOLOGY_BOILERPLATE_TERMS or st in _METHODOLOGY_BOILERPLATE_STEMS:
            continue
        subject_tokens.append(t)
        subject_stems.append(st)

    return subject_tokens, subject_stems


def _build_subject_vector(text: str, dim: int = 128) -> list[float]:
    _, stems = _extract_subject_tokens_and_stems(text, expand_acronyms=True)
    vec = [0.0] * dim
    if not stems:
        return vec
    for idx_pos, st in enumerate(stems):
        h = abs(hash(st)) % dim
        vec[h] += 1.0
        if idx_pos + 1 < len(stems):
            bigram = f"{st}_{stems[idx_pos + 1]}"
            hb = abs(hash(bigram)) % dim
            vec[hb] += 0.75
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [v / norm for v in vec]
    return vec


def _compute_zero_shot_similarity(source: dict, query_text: str, question_text: str = "") -> dict:
    title = str(source.get("title", "") or "")
    abstract = str(source.get("abstract") or source.get("content", "") or "")
    doc_text = f"{title} {abstract}".strip()

    anchor_text = question_text.strip() if question_text and question_text.strip() else query_text.strip()
    combined_query = f"{question_text} {query_text}".strip()

    _, anchor_stems = _extract_subject_tokens_and_stems(anchor_text)
    _, query_stems = _extract_subject_tokens_and_stems(combined_query)
    _, title_stems = _extract_subject_tokens_and_stems(title)
    _, doc_stems = _extract_subject_tokens_and_stems(doc_text)

    anchor_stem_set = set(anchor_stems)
    query_stem_set = set(query_stems)
    title_stem_set = set(title_stems)
    doc_stem_set = set(doc_stems)

    stem_counts = {}
    for st in anchor_stems:
        stem_counts[st] = stem_counts.get(st, 0) + 1
    for m in re.finditer(r"\b([a-z0-9]+)-([a-z0-9]+)\b", anchor_text.lower()):
        for part in (m.group(1), m.group(2)):
            pst = _stem_token(part)
            if pst in anchor_stem_set:
                stem_counts[pst] = stem_counts.get(pst, 0) + 1

    broad_context_stems = {_stem_token(w) for w in ("llm", "llms", "language", "model", "models", "natural", "processing")}
    specific_anchor_stems = anchor_stem_set - broad_context_stems

    anchor_overlap = anchor_stem_set & doc_stem_set
    specific_overlap = specific_anchor_stems & doc_stem_set
    query_overlap = query_stem_set & doc_stem_set
    title_overlap = query_stem_set & title_stem_set

    q_vec = _build_subject_vector(combined_query)
    d_vec = _build_subject_vector(doc_text)
    t_vec = _build_subject_vector(title)
    vec_sim = max(0.0, cosine_similarity(q_vec, d_vec))
    title_vec_sim = max(0.0, cosine_similarity(q_vec, t_vec))

    if isinstance(source.get("embedding"), list) and isinstance(source.get("query_embedding"), list):
        dense_sim = max(0.0, cosine_similarity(source["query_embedding"], source["embedding"]))
        vec_sim = 0.65 * dense_sim + 0.35 * vec_sim

    total_anchor_weight = sum(stem_counts.get(st, 1) for st in anchor_stem_set) or 1
    matched_anchor_weight = sum(stem_counts.get(st, 1) for st in anchor_overlap)
    anchor_coverage = matched_anchor_weight / total_anchor_weight

    return {
        "vec_sim": vec_sim,
        "title_vec_sim": title_vec_sim,
        "anchor_stem_set": anchor_stem_set,
        "specific_anchor_stems": specific_anchor_stems,
        "anchor_overlap": anchor_overlap,
        "specific_overlap": specific_overlap,
        "query_overlap": query_overlap,
        "title_overlap": title_overlap,
        "anchor_coverage": anchor_coverage,
        "stem_counts": stem_counts,
        "doc_stem_set": doc_stem_set,
    }


def _relevance_score(source, query_text, intent=None, content_status=None, question_text=""):
    sim_data = _compute_zero_shot_similarity(source, query_text, question_text=question_text)

    anchor_stem_set = sim_data["anchor_stem_set"]
    specific_anchor_stems = sim_data["specific_anchor_stems"]
    anchor_overlap = sim_data["anchor_overlap"]
    specific_overlap = sim_data["specific_overlap"]
    query_overlap = sim_data["query_overlap"]
    title_overlap = sim_data["title_overlap"]
    vec_sim = sim_data["vec_sim"]
    title_vec_sim = sim_data["title_vec_sim"]
    anchor_coverage = sim_data["anchor_coverage"]
    stem_counts = sim_data["stem_counts"]

    if anchor_stem_set and not anchor_overlap:
        return 0.0
    if not anchor_stem_set and not query_overlap:
        return 0.0
    if vec_sim < 0.08 and not title_overlap:
        return 0.0

    if specific_anchor_stems and not specific_overlap:
        return min(0.18, round(vec_sim * 0.4, 4))

    primary_focal_stems = {st for st, cnt in stem_counts.items() if cnt >= 2 and st in specific_anchor_stems}
    if primary_focal_stems and not (primary_focal_stems & sim_data["doc_stem_set"]):
        return min(0.20, round(vec_sim * 0.45, 4))

    title_ratio = len(title_overlap) / max(1, len(anchor_stem_set or sim_data["query_overlap"]))
    base_score = (0.45 * vec_sim) + (0.35 * anchor_coverage) + (0.20 * min(1.0, title_ratio + 0.3 * title_vec_sim))

    source_text = (
        f"{str(source.get('title', '') or '').lower()} "
        f"{str(source.get('abstract') or source.get('content', '') or '').lower()}"
    )
    combined_q = f"{question_text} {query_text}".lower()
    words = [w for w in re.findall(r"[a-z0-9]+", combined_q) if w not in STOPWORDS and len(w) > 2]
    query_phrases = set()
    for i in range(len(words) - 1):
        if words[i] not in _METHODOLOGY_BOILERPLATE_TERMS or words[i + 1] not in _METHODOLOGY_BOILERPLATE_TERMS:
            query_phrases.add(f"{words[i]} {words[i + 1]}")
    for i in range(len(words) - 2):
        query_phrases.add(f"{words[i]} {words[i + 1]} {words[i + 2]}")

    phrase_boost = 0.0
    if query_phrases:
        phrase_hits = sum(1 for phrase in query_phrases if phrase in source_text)
        if phrase_hits >= 2:
            phrase_boost = min(0.25, phrase_hits * 0.08)
        elif phrase_hits == 1:
            phrase_boost = 0.10

    return min(1.0, round(base_score + phrase_boost, 4))


def _relevance_reason(source, query_text, score, intent=None, question_text=""):
    sim_data = _compute_zero_shot_similarity(source, query_text, question_text=question_text)
    fields_of_study = [str(f).lower().strip() for f in (source.get("fields_of_study") or []) if str(f).strip()]

    if score <= 0.0:
        if fields_of_study:
            _, field_stems = _extract_subject_tokens_and_stems(" ".join(fields_of_study))
            if not (set(field_stems) & sim_data["anchor_stem_set"]):
                return "domain_mismatch_fields_of_study"
        return "domain_mismatch_penalty"

    specific_anchor_stems = sim_data["specific_anchor_stems"]
    specific_overlap = sim_data["specific_overlap"]
    stem_counts = sim_data["stem_counts"]
    primary_focal_stems = {st for st, cnt in stem_counts.items() if cnt >= 2 and st in specific_anchor_stems}

    if (specific_anchor_stems and not specific_overlap) or (
        primary_focal_stems and not (primary_focal_stems & sim_data["doc_stem_set"])
    ):
        if _stem_token("agent") in primary_focal_stems:
            return "general_llm_without_agent_focus"
        return "partial_subject_overlap"

    if score >= 0.55:
        if _stem_token("agent") in sim_data["anchor_overlap"]:
            return "multi_agent_comparative_match"
        if len(sim_data["title_overlap"]) >= 2:
            return "strong_title_match"
        return "strong_semantic_match"
    elif score >= 0.35:
        return "moderate_title_match" if sim_data["title_overlap"] else "moderate_semantic_match"
    elif score >= 0.15:
        return "abstract_match"
    return "generic_overlap_only"


def score_source(source, research_question, custom_quality_scores=None):
    """Score one source relative to one Planner question."""
    source_type = source.get("source_type", "other")
    query_texts = [research_question.get("question", "")]
    question_id = str(research_question.get("id", "")).strip().upper()

    for discovery in source.get("discovered_by", []):
        if discovery.get("question_id", "").upper() == question_id:
            query_texts.append(discovery.get("query_text", ""))

    best_relevance = 0.0
    best_relevance_reason = "no_query"
    content_status = str(source.get("content_status", "") or "").lower()
    question_text = research_question.get("question", "")
    for query in query_texts:
        q_score = _relevance_score(
            source,
            query,
            content_status=content_status,
            question_text=question_text
        )
        if q_score > best_relevance:
            best_relevance = q_score
            best_relevance_reason = _relevance_reason(
                source,
                query,
                q_score,
                question_text=question_text
            )

    if best_relevance == 0.0 and query_texts:
        best_relevance_reason = _relevance_reason(
            source,
            query_texts[0],
            0.0,
            question_text=question_text
        )

    relevance_score = best_relevance
    scores_dict = custom_quality_scores if custom_quality_scores is not None else SOURCE_QUALITY_SCORES
    source_quality_score = scores_dict.get(source_type, scores_dict.get("other", 0.40))
    planner_preference_score = _planner_preference_score(
        source_type,
        research_question.get("preferred_source_types", [])
    )
    tavily_score = _tavily_score(source)

    content_status_bonus = 0.0
    if content_status == "full":
        content_status_bonus = 0.15
    elif content_status == "partial":
        content_status_bonus = 0.10
    elif content_status == "snippet_only":
        content_status_bonus = 0.05
    elif content_status == "failed":
        content_status_bonus = -0.10

    requires_quantitative = bool(research_question.get("requires_quantitative_evidence", False))
    quantitative_bonus = 0.0
    if requires_quantitative:
        source_text_all = f"{source.get('title', '')} {source.get('abstract', '')} {source.get('content', '')}".lower()
        has_numbers = bool(re.search(r"\b\d+(?:\.\d+)?%|\b\d+\.\d+\b|\baccuracy\b|\bf1\b|\bpass@1\b|\bbaseline\b|\bscore\b", source_text_all))
        if has_numbers:
            quantitative_bonus = 0.08

    requires_counter = bool(research_question.get("requires_counter_evidence", False))
    counter_bonus = 0.0
    if requires_counter:
        is_counter_disc = any(
            str(d.get("query_type", "")).lower() == "counter"
            for d in source.get("discovered_by", [])
            if str(d.get("question_id", "")).upper() == question_id
        )
        source_text_all = f"{source.get('title', '')} {source.get('abstract', '')} {source.get('content', '')}".lower()
        has_counter_signals = any(
            kw in source_text_all
            for kw in ("limitation", "failure", "degradation", "overhead", "bottleneck", "trade-off", "underperform", "error propagation")
        )
        if is_counter_disc or has_counter_signals:
            counter_bonus = 0.08

    if relevance_score <= 0.05:
        final_score = relevance_score * RANKING_WEIGHTS["relevance"]
    elif relevance_score < 0.15:
        final_score = min(
            0.20,
            relevance_score * RANKING_WEIGHTS["relevance"]
            + source_quality_score * 0.05
        )
    else:
        authority_score = (
            source_quality_score * RANKING_WEIGHTS["source_quality"]
            + planner_preference_score * RANKING_WEIGHTS["planner_preference"]
            + tavily_score * RANKING_WEIGHTS["tavily"]
        )
        bonuses = content_status_bonus + quantitative_bonus + counter_bonus
        final_score = (
            relevance_score * RANKING_WEIGHTS["relevance"]
            + (authority_score + bonuses) * relevance_score
        )

    return {
        "relevance_score": round(relevance_score, 4),
        "relevance_reason": best_relevance_reason,
        "source_quality_score": round(source_quality_score, 4),
        "planner_preference_score": round(planner_preference_score, 4),
        "tavily_score": round(tavily_score, 4),
        "content_status_bonus": round(content_status_bonus, 4),
        "quantitative_bonus": round(quantitative_bonus, 4),
        "counter_bonus": round(counter_bonus, 4),
        "final_score": round(final_score, 4)
    }


def rank_sources(sources, research_questions, source_policy=None):
    """Classify and rank all canonical sources without filtering them."""
    questions_by_id = {
        str(question.get("id", "")).strip().upper(): question
        for question in research_questions
    }
    policy_scores = _build_policy_quality_scores(source_policy)

    for source in sources:
        source_type = classify_source(source)
        if source_type == "other" and source.get("source_type") in SOURCE_TYPES:
            source_type = source["source_type"]
        source["source_type"] = source_type if source_type in SOURCE_TYPES else "other"

        evidence_eligible, eligibility_reason = compute_evidence_eligibility(source, source_policy=source_policy)
        source["evidence_eligible"] = evidence_eligible
        source["eligibility_reason"] = eligibility_reason

        ranking_by_question = {}
        for question_id, question in questions_by_id.items():
            ranking_by_question[question_id] = score_source(
                source,
                question,
                custom_quality_scores=policy_scores
            )

        if ranking_by_question:
            best_question_id, best_score = max(
                ranking_by_question.items(),
                key=lambda item: (item[1]["final_score"], item[0])
            )
            source["ranking_score"] = best_score["final_score"]
            source["ranking_question_id"] = best_question_id
        else:
            source["ranking_score"] = 0.0
            source["ranking_question_id"] = ""

        source["ranking_by_question"] = ranking_by_question

    def _sort_key(s):
        score = float(s.get("ranking_score", 0.0) or 0.0)
        sid = str(s.get("source_id", "") or "")
        is_tavily = sid.startswith("S") and sid[1:].isdigit()
        num = int(sid[1:]) if is_tavily else 999999
        return (-score, num, sid)

    return sorted(sources, key=_sort_key)


def print_ranking_diagnostics(sources, research_questions, top_n=5):
    """Print concise per-question ranking diagnostics."""
    for question in research_questions:
        question_id = str(question.get("id", "")).strip().upper()
        ranked_sources = [
            source for source in sources
            if question_id in source.get("ranking_by_question", {})
        ]
        ranked_sources.sort(
            key=lambda source: source["ranking_by_question"][question_id]["final_score"],
            reverse=True
        )
        print(f"\n{question_id}\n\nQuestion:\n{question.get('question', '')}\n\nPlanner Preferred Types:")
        for preferred_type in question.get("preferred_source_types", []):
            print(f"- {preferred_type}")
        print("\nTOP RANKED SOURCES")
        for position, source in enumerate(ranked_sources[:top_n], start=1):
            scores = source["ranking_by_question"][question_id]
            provenance_types = sorted({
                discovery.get("query_type", "")
                for discovery in source.get("discovered_by", [])
                if discovery.get("question_id", "").strip().upper() == question_id
            })
            print(f"\n{position}. {source.get('source_id', '')}")
            print(f"   title: {source.get('title', '')}")
            print(f"   type: {source.get('source_type', 'other')}")
            print(f"   query provenance: {', '.join(provenance_types) or 'unknown'}")
            print(f"   relevance: {scores['relevance_score']:.4f}")
            print(f"   quality: {scores['source_quality_score']:.4f}")
            print(f"   preference: {scores['planner_preference_score']:.4f}")
            print(f"   tavily: {scores['tavily_score']:.4f}")
            print(f"   final: {scores['final_score']:.4f}")
