from google.genai import types


def build_planner_schema():
    """
    Return the JSON schema Gemini must follow when
    generating a Phase 1 research plan.

    This controls STRUCTURE.

    Logical consistency is still checked by the
    validation package after generation.
    """

    source_requirements_schema = types.Schema(
        type=types.Type.OBJECT,
        required=[
            "minimum_sources",
            "minimum_primary_sources",
            "minimum_quantitative_sources",
            "minimum_counter_evidence_sources",
        ],
        properties={
            "minimum_sources": types.Schema(
                type=types.Type.INTEGER,
            ),
            "minimum_primary_sources": types.Schema(
                type=types.Type.INTEGER,
            ),
            "minimum_quantitative_sources": types.Schema(
                type=types.Type.INTEGER,
            ),
            "minimum_counter_evidence_sources": types.Schema(
                type=types.Type.INTEGER,
            ),
        },
    )

    search_strategy_schema = types.Schema(
        type=types.Type.OBJECT,
        required=[
            "primary_queries",
            "secondary_queries",
            "counter_evidence_queries",
        ],
        properties={
            "primary_queries": _string_array_schema(),
            "secondary_queries": _string_array_schema(),
            "counter_evidence_queries":
                _string_array_schema(),
        },
    )

    question_schema = types.Schema(
        type=types.Type.OBJECT,
        required=[
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
        ],
        properties={
            "id": types.Schema(
                type=types.Type.STRING,
            ),

            "type": types.Schema(
                type=types.Type.STRING,
            ),

            "priority": types.Schema(
                type=types.Type.STRING,
                enum=[
                    "high",
                    "medium",
                    "low",
                ],
            ),

            "question": types.Schema(
                type=types.Type.STRING,
            ),

            "evidence_needed":
                _string_array_schema(),

            "search_strategy":
                search_strategy_schema,

            "preferred_source_types":
                _string_array_schema(),

            "source_requirements":
                source_requirements_schema,

            "requires_quantitative_evidence":
                types.Schema(
                    type=types.Type.BOOLEAN,
                ),

            "quantitative_fields":
                _string_array_schema(),

            "experimental_context_fields":
                _string_array_schema(),

            "requires_counter_evidence":
                types.Schema(
                    type=types.Type.BOOLEAN,
                ),

            "boundary_conditions":
                _string_array_schema(),

            "depends_on":
                _string_array_schema(),
        },
    )

    source_tier_schema = types.Schema(
        type=types.Type.OBJECT,
        required=[
            "tier",
            "source_types",
            "role",
        ],
        properties={
            "tier": types.Schema(
                type=types.Type.INTEGER,
            ),

            "source_types":
                _string_array_schema(),

            "role": types.Schema(
                type=types.Type.STRING,
                enum=[
                    "evidence",
                    "discovery",
                ],
            ),
        },
    )

    source_policy_schema = types.Schema(
        type=types.Type.OBJECT,
        required=[
            "tiers",
            "evidence_eligible_source_types",
            "discovery_only_source_types",
        ],
        properties={
            "tiers": types.Schema(
                type=types.Type.ARRAY,
                items=source_tier_schema,
            ),

            "evidence_eligible_source_types":
                _string_array_schema(),

            "discovery_only_source_types":
                _string_array_schema(),
        },
    )

    global_source_requirements_schema = (
        types.Schema(
            type=types.Type.OBJECT,
            required=[
                "minimum_total_sources",
                "minimum_primary_sources",
                "minimum_peer_reviewed_sources",
                "minimum_quantitative_sources",
                "minimum_counter_evidence_sources",
            ],
            properties={
                "minimum_total_sources":
                    types.Schema(
                        type=types.Type.INTEGER,
                    ),

                "minimum_primary_sources":
                    types.Schema(
                        type=types.Type.INTEGER,
                    ),

                "minimum_peer_reviewed_sources":
                    types.Schema(
                        type=types.Type.INTEGER,
                    ),

                "minimum_quantitative_sources":
                    types.Schema(
                        type=types.Type.INTEGER,
                    ),

                "minimum_counter_evidence_sources":
                    types.Schema(
                        type=types.Type.INTEGER,
                    ),
            },
        )
    )

    evidence_sufficiency_schema = types.Schema(
        type=types.Type.OBJECT,
        required=[
            "high",
            "medium",
            "low",
            "conflicting",
            "insufficient",
        ],
        properties={
            "high": types.Schema(
                type=types.Type.STRING,
            ),
            "medium": types.Schema(
                type=types.Type.STRING,
            ),
            "low": types.Schema(
                type=types.Type.STRING,
            ),
            "conflicting": types.Schema(
                type=types.Type.STRING,
            ),
            "insufficient": types.Schema(
                type=types.Type.STRING,
            ),
        },
    )

    stopping_criteria_schema = types.Schema(
        type=types.Type.OBJECT,
        required=[
            "require_minimum_total_sources",
            "require_minimum_primary_sources",
            "require_quantitative_evidence_when_applicable",
            "require_counter_evidence_when_applicable",
            "require_all_high_priority_questions_covered",
            "require_major_claims_supported",
            "require_contradictions_investigated",
            "maximum_search_rounds",
            "maximum_total_queries",
        ],
        properties={
            "require_minimum_total_sources":
                types.Schema(
                    type=types.Type.BOOLEAN,
                ),

            "require_minimum_primary_sources":
                types.Schema(
                    type=types.Type.BOOLEAN,
                ),

            "require_quantitative_evidence_when_applicable":
                types.Schema(
                    type=types.Type.BOOLEAN,
                ),

            "require_counter_evidence_when_applicable":
                types.Schema(
                    type=types.Type.BOOLEAN,
                ),

            "require_all_high_priority_questions_covered":
                types.Schema(
                    type=types.Type.BOOLEAN,
                ),

            "require_major_claims_supported":
                types.Schema(
                    type=types.Type.BOOLEAN,
                ),

            "require_contradictions_investigated":
                types.Schema(
                    type=types.Type.BOOLEAN,
                ),

            "maximum_search_rounds":
                types.Schema(
                    type=types.Type.INTEGER,
                ),

            "maximum_total_queries":
                types.Schema(
                    type=types.Type.INTEGER,
                ),
        },
    )

    claim_evidence_contract_schema = types.Schema(
        type=types.Type.OBJECT,
        required=[
            "required",
            "require_source_location",
            "require_support_or_contradict_label",
            "distinguish_observation_interpretation_conclusion",
        ],
        properties={
            "required": types.Schema(
                type=types.Type.BOOLEAN,
            ),

            "require_source_location":
                types.Schema(
                    type=types.Type.BOOLEAN,
                ),

            "require_support_or_contradict_label":
                types.Schema(
                    type=types.Type.BOOLEAN,
                ),

            "distinguish_observation_interpretation_conclusion":
                types.Schema(
                    type=types.Type.BOOLEAN,
                ),
        },
    )

    phase_2_output_contract_schema = types.Schema(
        type=types.Type.OBJECT,
        required=[
            "required_question_fields",
            "required_claim_fields",
            "required_source_fields",
            "required_final_fields",
        ],
        properties={
            "required_question_fields":
                _string_array_schema(),

            "required_claim_fields":
                _string_array_schema(),

            "required_source_fields":
                _string_array_schema(),

            "required_final_fields":
                _string_array_schema(),
        },
    )

    return types.Schema(
        type=types.Type.OBJECT,

        required=[
            "topic",
            "domain",
            "research_mode",
            "temporal_sensitivity",
            "hypothesis",
            "competing_hypotheses",
            "falsification_criteria",
            "evaluation_criteria",
            "source_policy",
            "global_source_requirements",
            "source_diversity_requirements",
            "questions",
            "evidence_sufficiency",
            "stopping_criteria",
            "claim_evidence_contract",
            "allowed_conclusion_statuses",
            "phase_2_output_contract",
        ],

        properties={
            "topic": types.Schema(
                type=types.Type.STRING,
            ),

            "domain": types.Schema(
                type=types.Type.STRING,
            ),

            "research_mode": types.Schema(
                type=types.Type.STRING,
                enum=[
                    "exploratory",
                    "comparative",
                    "causal",
                    "evaluative",
                    "descriptive",
                ],
            ),

            "temporal_sensitivity": types.Schema(
                type=types.Type.STRING,
                enum=[
                    "low",
                    "medium",
                    "high",
                ],
            ),

            # Gemini structured output schemas can become
            # troublesome when trying to represent
            # string | null across SDK/model versions.
            #
            # Therefore the model returns an empty string
            # when no hypothesis is applicable.
            #
            # plan_validator.py will normalize "" -> None.
            "hypothesis": types.Schema(
                type=types.Type.STRING,
            ),

            "competing_hypotheses":
                _string_array_schema(),

            "falsification_criteria":
                _string_array_schema(),

            "evaluation_criteria":
                _string_array_schema(),

            "source_policy":
                source_policy_schema,

            "global_source_requirements":
                global_source_requirements_schema,

            "source_diversity_requirements":
                _string_array_schema(),

            "questions": types.Schema(
                type=types.Type.ARRAY,
                items=question_schema,
            ),

            "evidence_sufficiency":
                evidence_sufficiency_schema,

            "stopping_criteria":
                stopping_criteria_schema,

            "claim_evidence_contract":
                claim_evidence_contract_schema,

            "allowed_conclusion_statuses":
                _string_array_schema(),

            "phase_2_output_contract":
                phase_2_output_contract_schema,
        },
    )


def _string_array_schema():
    """
    Convenience schema for list[str].
    """

    return types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.STRING,
        ),
    )