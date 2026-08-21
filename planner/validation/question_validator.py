from ..constants import (
    ALLOWED_PRIORITIES,
    MIN_QUESTIONS,
    MAX_QUESTIONS,
)

from .helpers import (
    clean_string,
    clean_string_list,
    require_bool,
    require_dict,
    require_list,
    require_non_negative_int,
)


REQUIRED_QUESTION_FIELDS = {
    "id",
    "type",
    "priority",
    "question",
    "evidence_needed",
    "search_strategy",
    "preferred_source_types",
    "source_requirements",
    "requires_quantitative_evidence",
    "quantitative_fields",
    "experimental_context_fields",
    "requires_counter_evidence",
    "boundary_conditions",
    "depends_on",
}


def validate_questions(value):
    """
    Validate all Planner research questions.
    """

    questions = require_list(
        value,
        "questions",
    )

    if not (
        MIN_QUESTIONS
        <= len(questions)
        <= MAX_QUESTIONS
    ):
        raise RuntimeError(
            "Planner must generate between "
            f"{MIN_QUESTIONS} and "
            f"{MAX_QUESTIONS} questions."
        )

    cleaned_questions = []
    seen_ids = set()

    for index, item in enumerate(
        questions,
        start=1,
    ):
        cleaned_question = _validate_question(
            item=item,
            index=index,
            seen_ids=seen_ids,
        )

        cleaned_questions.append(
            cleaned_question
        )

    _require_high_priority_question(
        cleaned_questions
    )

    _validate_dependencies(
        cleaned_questions
    )

    return cleaned_questions


def _validate_question(
    item,
    index,
    seen_ids,
):
    require_dict(
        item,
        f"Question {index}",
    )

    missing = (
        REQUIRED_QUESTION_FIELDS
        - item.keys()
    )

    if missing:
        raise RuntimeError(
            f"Question {index} is missing: "
            + ", ".join(sorted(missing))
        )

    # ======================================================
    # ID
    # ======================================================

    question_id = clean_string(
        item["id"]
    ).upper()

    if not question_id:
        raise RuntimeError(
            f"Question {index} has no ID."
        )

    if question_id in seen_ids:
        raise RuntimeError(
            f"Duplicate question ID: {question_id}"
        )

    seen_ids.add(question_id)

    # ======================================================
    # TYPE
    # ======================================================

    question_type = clean_string(
        item["type"]
    ).lower()

    if not question_type:
        raise RuntimeError(
            f"{question_id} has no type."
        )

    # ======================================================
    # PRIORITY
    # ======================================================

    priority = clean_string(
        item["priority"]
    ).lower()

    if priority not in ALLOWED_PRIORITIES:
        raise RuntimeError(
            f"{question_id} has invalid "
            f"priority: {priority}"
        )

    # ======================================================
    # QUESTION TEXT
    # ======================================================

    question_text = clean_string(
        item["question"]
    )

    if not question_text:
        raise RuntimeError(
            f"{question_id} is empty."
        )

    # ======================================================
    # EVIDENCE NEEDED
    # ======================================================

    evidence_needed = clean_string_list(
        item["evidence_needed"],
        f"{question_id}.evidence_needed",
    )

    if not evidence_needed:
        raise RuntimeError(
            f"{question_id} must specify "
            "evidence requirements."
        )

    # ======================================================
    # SEARCH STRATEGY
    # ======================================================

    (
        primary_queries,
        secondary_queries,
        counter_queries,
    ) = _validate_search_strategy(
        question_id,
        item["search_strategy"],
    )

    # ======================================================
    # PREFERRED SOURCE TYPES
    # ======================================================

    preferred_source_types = (
        clean_string_list(
            item["preferred_source_types"],
            (
                f"{question_id}."
                "preferred_source_types"
            ),
        )
    )

    if not preferred_source_types:
        raise RuntimeError(
            f"{question_id} must specify "
            "preferred source types."
        )

    # ======================================================
    # SOURCE REQUIREMENTS
    # ======================================================

    source_requirements = (
        _validate_source_requirements(
            question_id,
            item["source_requirements"],
        )
    )

    # ======================================================
    # QUANTITATIVE EVIDENCE
    # ======================================================

    requires_quantitative = require_bool(
        item[
            "requires_quantitative_evidence"
        ],
        (
            f"{question_id}."
            "requires_quantitative_evidence"
        ),
    )

    quantitative_fields = clean_string_list(
        item["quantitative_fields"],
        f"{question_id}.quantitative_fields",
    )

    if (
        requires_quantitative
        and not quantitative_fields
    ):
        raise RuntimeError(
            f"{question_id} requires "
            "quantitative evidence but "
            "quantitative_fields is empty."
        )

    # If quantitative evidence is not required,
    # normalize both the fields and minimum count.
    if not requires_quantitative:
        quantitative_fields = []

        source_requirements[
            "minimum_quantitative_sources"
        ] = 0

    # ======================================================
    # EXPERIMENTAL CONTEXT
    # ======================================================

    experimental_context_fields = (
        clean_string_list(
            item[
                "experimental_context_fields"
            ],
            (
                f"{question_id}."
                "experimental_context_fields"
            ),
        )
    )

    # ======================================================
    # COUNTER EVIDENCE
    # ======================================================

    requires_counter = require_bool(
        item["requires_counter_evidence"],
        (
            f"{question_id}."
            "requires_counter_evidence"
        ),
    )

    if (
        requires_counter
        and not counter_queries
    ):
        raise RuntimeError(
            f"{question_id} requires "
            "counter-evidence but has no "
            "counter-evidence query."
        )

    # If counter-evidence is not required,
    # normalize the queries and minimum count.
    if not requires_counter:
        counter_queries = []

        source_requirements[
            "minimum_counter_evidence_sources"
        ] = 0

    # ======================================================
    # BOUNDARY CONDITIONS
    # ======================================================

    boundary_conditions = clean_string_list(
        item["boundary_conditions"],
        f"{question_id}.boundary_conditions",
    )

    # ======================================================
    # DEPENDENCIES
    # ======================================================

    depends_on = [
        dependency.upper()
        for dependency in clean_string_list(
            item["depends_on"],
            f"{question_id}.depends_on",
        )
    ]

    return {
        "id": question_id,
        "type": question_type,
        "priority": priority,
        "question": question_text,
        "evidence_needed": evidence_needed,

        "search_strategy": {
            "primary_queries":
                primary_queries,
            "secondary_queries":
                secondary_queries,
            "counter_evidence_queries":
                counter_queries,
        },

        "preferred_source_types":
            preferred_source_types,

        "source_requirements":
            source_requirements,

        "requires_quantitative_evidence":
            requires_quantitative,

        "quantitative_fields":
            quantitative_fields,

        "experimental_context_fields":
            experimental_context_fields,

        "requires_counter_evidence":
            requires_counter,

        "boundary_conditions":
            boundary_conditions,

        "depends_on":
            depends_on,
    }


def _validate_search_strategy(
    question_id,
    value,
):
    search_strategy = require_dict(
        value,
        f"{question_id}.search_strategy",
    )

    required_fields = {
        "primary_queries",
        "secondary_queries",
        "counter_evidence_queries",
    }

    missing = (
        required_fields
        - search_strategy.keys()
    )

    if missing:
        raise RuntimeError(
            f"{question_id}.search_strategy "
            "is missing: "
            + ", ".join(sorted(missing))
        )

    primary_queries = clean_string_list(
        search_strategy["primary_queries"],
        (
            f"{question_id}.search_strategy."
            "primary_queries"
        ),
    )

    secondary_queries = clean_string_list(
        search_strategy["secondary_queries"],
        (
            f"{question_id}.search_strategy."
            "secondary_queries"
        ),
    )

    counter_queries = clean_string_list(
        search_strategy[
            "counter_evidence_queries"
        ],
        (
            f"{question_id}.search_strategy."
            "counter_evidence_queries"
        ),
    )

    if not primary_queries:
        raise RuntimeError(
            f"{question_id} must have at "
            "least one primary query."
        )

    return (
        primary_queries,
        secondary_queries,
        counter_queries,
    )


def _validate_source_requirements(
    question_id,
    value,
):
    requirements = require_dict(
        value,
        f"{question_id}.source_requirements",
    )

    required_fields = {
        "minimum_sources",
        "minimum_primary_sources",
        "minimum_quantitative_sources",
        "minimum_counter_evidence_sources",
    }

    missing = (
        required_fields
        - requirements.keys()
    )

    if missing:
        raise RuntimeError(
            f"{question_id}.source_requirements "
            "is missing: "
            + ", ".join(sorted(missing))
        )

    cleaned = {}

    for field in required_fields:
        cleaned[field] = (
            require_non_negative_int(
                requirements[field],
                (
                    f"{question_id}."
                    "source_requirements."
                    f"{field}"
                ),
            )
        )

    minimum_sources = cleaned[
        "minimum_sources"
    ]

    if minimum_sources <= 0:
        raise RuntimeError(
            f"{question_id}.minimum_sources "
            "must be greater than zero."
        )

    for field in (
        "minimum_primary_sources",
        "minimum_quantitative_sources",
        "minimum_counter_evidence_sources",
    ):
        if cleaned[field] > minimum_sources:
            raise RuntimeError(
                f"{question_id}.{field} "
                "cannot exceed minimum_sources."
            )

    return cleaned


def _require_high_priority_question(
    questions,
):
    if not any(
        question["priority"] == "high"
        for question in questions
    ):
        raise RuntimeError(
            "Planner must generate at least one "
            "high-priority question."
        )


def _validate_dependencies(questions):
    """
    Validate references and detect dependency cycles.
    """

    valid_ids = {
        question["id"]
        for question in questions
    }

    dependency_graph = {}

    for question in questions:
        question_id = question["id"]

        dependencies = set(
            question["depends_on"]
        )

        dependency_graph[
            question_id
        ] = dependencies

        for dependency in dependencies:
            if dependency not in valid_ids:
                raise RuntimeError(
                    f"{question_id} depends on "
                    "unknown question "
                    f"{dependency}."
                )

            if dependency == question_id:
                raise RuntimeError(
                    f"{question_id} cannot "
                    "depend on itself."
                )

    _detect_circular_dependencies(
        dependency_graph
    )


def _detect_circular_dependencies(
    dependency_graph,
):
    visiting = set()
    visited = set()

    def visit(question_id):
        if question_id in visiting:
            raise RuntimeError(
                "Circular question dependency "
                f"detected at {question_id}."
            )

        if question_id in visited:
            return

        visiting.add(question_id)

        for dependency in (
            dependency_graph[question_id]
        ):
            visit(dependency)

        visiting.remove(question_id)
        visited.add(question_id)

    for question_id in dependency_graph:
        visit(question_id)