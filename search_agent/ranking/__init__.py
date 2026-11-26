"""
search_agent/ranking/__init__.py

Modularized source classification, policy normalization, and zero-shot scoring package.
"""

from .policy import (
    EVIDENCE_ELIGIBLE_SOURCE_TYPES,
    EVIDENCE_EXCLUSION_REASONS,
    PREFERENCE_TO_SOURCE_TYPES,
    RANKING_WEIGHTS,
    SOURCE_QUALITY_SCORES,
    SOURCE_TYPES,
    _build_policy_quality_scores,
    _domain_and_text,
    _normalize_policy_type_to_source_types,
    _planner_preference_score,
    _tavily_score,
    classify_source,
    compute_evidence_eligibility,
)
from .scorer import (
    STOPWORDS,
    _ACRONYM_EXPANSIONS,
    _METHODOLOGY_BOILERPLATE_STEMS,
    _METHODOLOGY_BOILERPLATE_TERMS,
    _build_subject_vector,
    _compute_zero_shot_similarity,
    _extract_subject_tokens_and_stems,
    _relevance_reason,
    _relevance_score,
    _stem_token,
    _tokens,
    print_ranking_diagnostics,
    rank_sources,
    score_source,
)

__all__ = [
    "SOURCE_TYPES",
    "SOURCE_QUALITY_SCORES",
    "RANKING_WEIGHTS",
    "EVIDENCE_ELIGIBLE_SOURCE_TYPES",
    "EVIDENCE_EXCLUSION_REASONS",
    "PREFERENCE_TO_SOURCE_TYPES",
    "STOPWORDS",
    "classify_source",
    "compute_evidence_eligibility",
    "score_source",
    "rank_sources",
    "print_ranking_diagnostics",
    "_build_policy_quality_scores",
    "_relevance_score",
    "_relevance_reason",
    "_compute_zero_shot_similarity",
]
