from ..constants import (
    MAX_SEARCH_ROUNDS,
    MAX_TOTAL_QUERIES,
)

from .helpers import (
    clean_string,
    require_bool,
    require_dict,
    require_positive_int,
)


def validate_evidence_sufficiency(value):
    """
    Validate evidence sufficiency definitions.
    """

    evidence_sufficiency = require_dict(
        value,
        "evidence_sufficiency",
    )

    required_fields = {
        "high",
        "medium",
        "low",
        "conflicting",
        "insufficient",
    }

    missing = (
        required_fields
        - evidence_sufficiency.keys()
    )

    if missing:
        raise RuntimeError(
            "evidence_sufficiency is missing: "
            + ", ".join(sorted(missing))
        )

    cleaned = {}

    for field in required_fields:
        description = clean_string(
            evidence_sufficiency[field]
        )

        if not description:
            raise RuntimeError(
                "Evidence sufficiency "
                f"'{field}' cannot be empty."
            )

        cleaned[field] = description

    return cleaned


def validate_stopping_criteria(value):
    """
    Validate stopping rules and enforce
    hard research-budget limits.
    """

    stopping_criteria = require_dict(
        value,
        "stopping_criteria",
    )

    boolean_fields = {
        "require_minimum_total_sources",
        "require_minimum_primary_sources",
        "require_quantitative_evidence_when_applicable",
        "require_counter_evidence_when_applicable",
        "require_all_high_priority_questions_covered",
        "require_major_claims_supported",
        "require_contradictions_investigated",
    }

    required_fields = (
        boolean_fields
        | {
            "maximum_search_rounds",
            "maximum_total_queries",
        }
    )

    missing = (
        required_fields
        - stopping_criteria.keys()
    )

    if missing:
        raise RuntimeError(
            "stopping_criteria is missing: "
            + ", ".join(sorted(missing))
        )

    cleaned = {}

    for field in boolean_fields:
        cleaned[field] = require_bool(
            stopping_criteria[field],
            f"stopping_criteria.{field}",
        )

    maximum_search_rounds = (
        require_positive_int(
            stopping_criteria[
                "maximum_search_rounds"
            ],
            (
                "stopping_criteria."
                "maximum_search_rounds"
            ),
        )
    )

    maximum_total_queries = (
        require_positive_int(
            stopping_criteria[
                "maximum_total_queries"
            ],
            (
                "stopping_criteria."
                "maximum_total_queries"
            ),
        )
    )

    cleaned[
        "maximum_search_rounds"
    ] = min(
        maximum_search_rounds,
        MAX_SEARCH_ROUNDS,
    )

    cleaned[
        "maximum_total_queries"
    ] = min(
        maximum_total_queries,
        MAX_TOTAL_QUERIES,
    )

    return cleaned