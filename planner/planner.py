import json


# ==========================================================
# ALLOWED VALUES
# ==========================================================

ALLOWED_PRIORITIES = {
    "high",
    "medium",
    "low",
}

ALLOWED_RESEARCH_MODES = {
    "exploratory",
    "comparative",
    "causal",
    "evaluative",
    "descriptive",
}

ALLOWED_TEMPORAL_SENSITIVITY = {
    "low",
    "medium",
    "high",
}

ALLOWED_SOURCE_ROLES = {
    "evidence",
    "discovery",
}

ALLOWED_CONCLUSION_STATUSES = {
    "supported",
    "partially_supported",
    "not_supported",
    "conflicting",
    "insufficient_evidence",
}


# ==========================================================
# LIMITS
# ==========================================================

MIN_QUESTIONS = 3
MAX_QUESTIONS = 5

MAX_SEARCH_ROUNDS = 3
MAX_TOTAL_QUERIES = 20


# ==========================================================
# VALIDATION HELPERS
# ==========================================================

def clean_string(value):
    """
    Convert a value into stripped text.
    """

    if value is None:
        return ""

    return str(value).strip()


def require_dict(value, field_name):
    """
    Require a JSON object.
    """

    if not isinstance(value, dict):
        raise RuntimeError(
            f"'{field_name}' must be a JSON object."
        )

    return value


def require_list(value, field_name):
    """
    Require a JSON list.
    """

    if not isinstance(value, list):
        raise RuntimeError(
            f"'{field_name}' must be a list."
        )

    return value


def clean_string_list(value, field_name):
    """
    Validate and clean a list of strings.
    """

    require_list(
        value,
        field_name
    )

    cleaned = []

    for item in value:

        text = clean_string(item)

        if text:
            cleaned.append(text)

    return cleaned


def require_bool(value, field_name):
    """
    Require a Boolean.
    """

    if not isinstance(value, bool):
        raise RuntimeError(
            f"'{field_name}' must be true or false."
        )

    return value


def require_non_negative_int(
    value,
    field_name
):
    """
    Require an integer >= 0.
    """

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise RuntimeError(
            f"'{field_name}' must be a "
            "non-negative integer."
        )

    return value


def require_positive_int(
    value,
    field_name
):
    """
    Require an integer > 0.
    """

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise RuntimeError(
            f"'{field_name}' must be a "
            "positive integer."
        )

    return value


# ==========================================================
# PLANNER AGENT
# ==========================================================

def create_research_plan(
    client,
    model,
    topic
):
    """
    Create a machine-executable research plan.

    The Planner decides:

    - research domain
    - research mode
    - temporal sensitivity
    - neutral hypothesis
    - competing hypotheses
    - falsification criteria
    - evaluation criteria
    - research questions
    - search queries
    - source preferences
    - minimum source requirements
    - quantitative requirements
    - experimental context
    - counter-evidence requirements
    - boundary conditions
    - dependencies
    - source hierarchy
    - source diversity
    - evidence sufficiency
    - stopping criteria
    - claim/evidence requirements
    - Phase 2 output contract

    The Planner DOES NOT:

    - search the web
    - find actual sources
    - provide URLs
    - rank actual URLs
    - extract evidence
    - verify citations
    - answer the research topic
    - produce the final conclusion
    """

    topic = clean_string(topic)

    if not topic:
        raise ValueError(
            "Research topic cannot be empty."
        )

    # ======================================================
    # PROMPT
    # ======================================================

    prompt = f"""
You are the Planner Agent in an autonomous research
system.

Your job is to create a rigorous MACHINE-EXECUTABLE
research strategy.

You are NOT the Search Agent.
You are NOT the Research Agent.
You are NOT the Synthesis Agent.

Do NOT answer the research topic.

Do NOT search the web.

Do NOT invent:

- sources
- URLs
- paper names
- authors
- statistics
- benchmark results
- research findings

Your output will be executed programmatically by
another component.

Therefore all important research instructions must
be represented explicitly in structured JSON.


RESEARCH TOPIC:

{topic}


==================================================
1. RESEARCH DOMAIN
==================================================

Determine the primary domain.

Examples:

AI/ML
Medical
Economics
Technology
Environment
Science
Law
Social Science
General

Another concise domain may be used when appropriate.


==================================================
2. RESEARCH MODE
==================================================

Choose exactly one:

exploratory
comparative
causal
evaluative
descriptive

exploratory:
Broad investigation.

comparative:
Comparison between alternatives.

causal:
Investigation of causal influence.

evaluative:
Assessment of effectiveness, performance,
reliability, risk, usefulness, or outcomes.

descriptive:
Description of current state or characteristics.


==================================================
3. TEMPORAL SENSITIVITY
==================================================

Choose exactly one:

low
medium
high

LOW:
Older strong evidence can remain highly useful.

MEDIUM:
Recent evidence matters, but foundational evidence
is also important.

HIGH:
The subject changes rapidly and recent evidence is
particularly important.

Do not assume older evidence is automatically weak.


==================================================
4. WORKING HYPOTHESIS
==================================================

For comparative, causal, or evaluative research,
create a neutral working hypothesis.

The hypothesis must be:

- testable
- neutral
- falsifiable or qualifiable
- open to support
- open to contradiction

It must NOT assume the conclusion.

For exploratory/descriptive investigations where
a hypothesis would be artificial, return null.


==================================================
5. COMPETING HYPOTHESES
==================================================

When useful, formulate reasonable competing
hypotheses.

They should represent genuinely different possible
outcomes.

Examples:

- the proposed method improves the outcome
- it does not materially improve the outcome
- it improves some dimensions but creates new
  failure modes
- effectiveness depends strongly on specific
  conditions

Return [] when not appropriate.


==================================================
6. FALSIFICATION CRITERIA
==================================================

When a working hypothesis exists, specify observable
evidence that would materially weaken, contradict,
or reject it.

Do not invent findings.

Return [] when not applicable.


==================================================
7. EVALUATION CRITERIA
==================================================

Identify ambiguous or broad concepts requiring
operational interpretation.

Examples:

effectiveness
reliability
safety
scalability
economic usefulness
performance
fairness

Define measurable or observable criteria that
later research can use.

Return [] if none are necessary.


==================================================
8. RESEARCH QUESTIONS
==================================================

Generate 3 to 5 research questions.

Each question must investigate a distinct important
dimension of the topic.

Possible question types:

definition
background
current_state
performance
comparison
quantitative
causes
effects
risks
limitations
boundary_conditions
counter_evidence
economics
social_impact
production
future

Do not force every topic into the same categories.

Questions must form a coherent investigation.


==================================================
9. PRIORITY
==================================================

Each question must have:

high
medium
or
low

At least one question must be HIGH.

HIGH:
Essential to answering the topic.

MEDIUM:
Important supporting investigation.

LOW:
Secondary context.


==================================================
10. EVIDENCE REQUIREMENTS
==================================================

For each question specify the evidence required to
answer it.

For AI/ML this may include:

- model
- dataset
- benchmark
- baseline
- metric
- experimental setup
- accuracy
- hallucination rate
- latency
- cost
- ablation evidence
- retrieval configuration
- reproducibility information

For medicine it may include:

- study design
- population
- sample size
- intervention
- outcome
- effect size
- confidence interval

For economics it may include:

- official statistics
- time period
- economic indicator
- comparison baseline
- cost
- productivity
- employment data

Use requirements appropriate to the actual topic.


==================================================
11. SEARCH STRATEGY
==================================================

For every question generate:

primary_queries
secondary_queries
counter_evidence_queries

Primary queries:
Direct searches for evidence.

Secondary queries:
Alternative terminology, synonyms, technical
phrasing, specific mechanisms, limitations, or
failure modes.

Counter-evidence queries:
Searches designed to challenge or qualify likely
interpretations.

Queries must be concise search-engine-friendly
queries.

Do NOT include URLs.

Do NOT claim the searches were executed.


==================================================
12. SOURCE QUALITY HIERARCHY
==================================================

Create a source hierarchy appropriate to the domain.

For AI/ML a reasonable hierarchy might include:

Tier 1:
peer-reviewed primary research

Tier 2:
conference papers

Tier 3:
high-quality primary research preprints

Tier 4:
systematic reviews or survey papers

Tier 5:
official technical reports or documentation

Tier 6:
industry research

Tier 7:
technical journalism or technical blogs

Tier 8:
general blogs, forums, social media, videos

This hierarchy must be adapted to the domain.

Every tier has a role:

evidence
or
discovery

EVIDENCE:
May directly support research claims.

DISCOVERY:
Useful for finding stronger evidence but should not
normally satisfy evidence requirements.

Do NOT provide actual sources.


==================================================
13. GLOBAL SOURCE REQUIREMENTS
==================================================

Define:

minimum_total_sources
minimum_primary_sources
minimum_peer_reviewed_sources
minimum_quantitative_sources
minimum_counter_evidence_sources

Requirements must be realistic for the topic.

Do not blindly use the same values for every topic.

Do not make the requirements unnecessarily large.


==================================================
14. QUESTION-SPECIFIC SOURCE REQUIREMENTS
==================================================

For every question define:

minimum_sources
minimum_primary_sources
minimum_quantitative_sources
minimum_counter_evidence_sources

High-priority questions should generally require
stronger coverage.

Values must be internally consistent.


==================================================
15. QUANTITATIVE EVIDENCE
==================================================

For each question determine:

requires_quantitative_evidence

If true, specify quantitative_fields.

Possible AI/ML fields include:

model
dataset
task
baseline
experimental_method
metric
baseline_score
experimental_score
absolute_difference
relative_difference
latency
cost
compute
sample_size
location_in_source

Only request relevant fields.

If quantitative evidence is false:

quantitative_fields must be [].

Missing numbers must later remain missing.

They must never be estimated or invented.


==================================================
16. EXPERIMENTAL CONTEXT
==================================================

When empirical results require context, specify
experimental_context_fields.

For AI/ML:

model
model size
dataset
task
retriever
embedding model
top-k
baseline
evaluation metric
experimental setup

For medicine:

population
sample size
dosage
duration
control
outcome measure

For economics:

country
time period
population
industry
baseline
economic conditions

Return [] when not relevant.


==================================================
17. COUNTER-EVIDENCE
==================================================

Determine whether each question requires explicit
counter-evidence research.

Use true for questions involving:

- effectiveness
- comparison
- causality
- strong performance claims
- risks
- controversial conclusions

If true, provide at least one counter-evidence
search query.


==================================================
18. BOUNDARY CONDITIONS
==================================================

For every question identify conditions under which
the answer may change.

Examples for AI/ML:

model
dataset
benchmark
retrieval quality
architecture
evaluation methodology
compute budget

Return [] if no meaningful conditions exist.


==================================================
19. QUESTION DEPENDENCIES
==================================================

Specify logical dependencies using question IDs.

Examples:

Q1: []
Q2: ["Q1"]

Do not create dependencies unnecessarily.

Questions must not depend on themselves.

Do not create circular dependencies.


==================================================
20. SOURCE DIVERSITY
==================================================

Specify relevant dimensions across which evidence
should be diverse.

Examples:

authors
institutions
research groups
datasets
models
benchmarks
journals
conferences
populations
geographic regions
time periods

Only include relevant dimensions.


==================================================
21. EVIDENCE SUFFICIENCY
==================================================

Use these evidence levels:

high
medium
low
conflicting
insufficient

Define each one.

General interpretation:

high:
Multiple strong independent sources agree.

medium:
Useful evidence exists but coverage, independence,
or quality is limited.

low:
Weak, indirect, or sparse evidence.

conflicting:
Comparable evidence supports materially different
conclusions.

insufficient:
Evidence cannot justify a conclusion.


==================================================
22. STOPPING CRITERIA
==================================================

Research should be considered sufficient only when
appropriate requirements are satisfied.

Include:

require_minimum_total_sources
require_minimum_primary_sources
require_quantitative_evidence_when_applicable
require_counter_evidence_when_applicable
require_all_high_priority_questions_covered
require_major_claims_supported
require_contradictions_investigated

Also define:

maximum_search_rounds
maximum_total_queries

maximum_search_rounds must not exceed:
{MAX_SEARCH_ROUNDS}

maximum_total_queries must not exceed:
{MAX_TOTAL_QUERIES}

Research can stop because:

1. evidence is sufficient

OR

2. the research budget is exhausted

Budget exhaustion must NOT be treated as evidence
sufficiency.


==================================================
23. CLAIM-EVIDENCE CONTRACT
==================================================

Later research should map:

Claim
-> Evidence
-> Source
-> Source location
-> Relationship
-> Confidence

Require claim-level traceability.

Require the later system to distinguish:

OBSERVATION:
What a source directly reports.

INTERPRETATION:
What that evidence may suggest.

CONCLUSION:
What can reasonably be inferred across evidence.


==================================================
24. FINAL CONCLUSION STATUS
==================================================

The final system must be allowed to return:

supported
partially_supported
not_supported
conflicting
insufficient_evidence

It must never be forced to produce a confident
answer when evidence is insufficient.


==================================================
25. PHASE 2 OUTPUT CONTRACT
==================================================

Define the fields Phase 2 must eventually return.

Question-level output must include:

id
status
claims
quantitative_evidence
limitations

Claim output must include:

claim_id
claim
supporting_source_ids
contradicting_source_ids
confidence

Source output must include:

source_id
title
url
source_type
publication_date
evidence_eligible

Final output must include:

unanswered_questions
stopping_criteria
overall_conclusion


==================================================
GENERAL REQUIREMENTS
==================================================

1. Generate 3 to 5 questions.

2. Do NOT answer the topic.

3. Do NOT provide research findings.

4. Do NOT fabricate citations.

5. Do NOT fabricate source names.

6. Do NOT fabricate URLs.

7. Do NOT claim evidence has already been found.

8. Avoid duplicate questions.

9. Search queries must be concise.

10. Source requirements must be realistic.

11. Priority must be exactly:
    high, medium, or low.

12. Return ONLY valid JSON.

13. Do not include Markdown.

14. Do not include ```json.

15. Do not include text before or after the JSON.


==================================================
REQUIRED JSON STRUCTURE
==================================================

{{
    "topic": "{topic}",

    "domain": "AI/ML",

    "research_mode": "evaluative",

    "temporal_sensitivity": "high",

    "hypothesis": "Neutral hypothesis or null",

    "competing_hypotheses": [
        "Alternative hypothesis"
    ],

    "falsification_criteria": [
        "Observable falsification criterion"
    ],

    "evaluation_criteria": [
        "Operational criterion"
    ],

    "source_policy": {{
        "tiers": [
            {{
                "tier": 1,
                "source_types": [
                    "peer-reviewed primary research"
                ],
                "role": "evidence"
            }},
            {{
                "tier": 2,
                "source_types": [
                    "conference paper"
                ],
                "role": "evidence"
            }},
            {{
                "tier": 3,
                "source_types": [
                    "technical blog",
                    "video"
                ],
                "role": "discovery"
            }}
        ],

        "evidence_eligible_source_types": [
            "peer-reviewed primary research",
            "conference paper"
        ],

        "discovery_only_source_types": [
            "technical blog",
            "video"
        ]
    }},

    "global_source_requirements": {{
        "minimum_total_sources": 10,
        "minimum_primary_sources": 4,
        "minimum_peer_reviewed_sources": 3,
        "minimum_quantitative_sources": 3,
        "minimum_counter_evidence_sources": 2
    }},

    "source_diversity_requirements": [
        "authors",
        "institutions",
        "datasets"
    ],

    "questions": [
        {{
            "id": "Q1",

            "type": "performance",

            "priority": "high",

            "question": "Focused research question?",

            "evidence_needed": [
                "Required evidence"
            ],

            "search_strategy": {{
                "primary_queries": [
                    "primary search query"
                ],

                "secondary_queries": [
                    "secondary search query"
                ],

                "counter_evidence_queries": [
                    "counter evidence query"
                ]
            }},

            "preferred_source_types": [
                "peer-reviewed primary research"
            ],

            "source_requirements": {{
                "minimum_sources": 3,
                "minimum_primary_sources": 2,
                "minimum_quantitative_sources": 2,
                "minimum_counter_evidence_sources": 1
            }},

            "requires_quantitative_evidence": true,

            "quantitative_fields": [
                "model",
                "dataset",
                "metric",
                "baseline_score",
                "experimental_score",
                "location_in_source"
            ],

            "experimental_context_fields": [
                "model",
                "dataset",
                "task",
                "evaluation metric"
            ],

            "requires_counter_evidence": true,

            "boundary_conditions": [
                "model",
                "dataset"
            ],

            "depends_on": []
        }}
    ],

    "evidence_sufficiency": {{
        "high": "Multiple independent strong sources agree.",
        "medium": "Useful evidence exists but coverage or quality is limited.",
        "low": "Evidence is weak, indirect, or sparse.",
        "conflicting": "Comparable evidence supports materially different conclusions.",
        "insufficient": "Evidence cannot justify a conclusion."
    }},

    "stopping_criteria": {{
        "require_minimum_total_sources": true,
        "require_minimum_primary_sources": true,
        "require_quantitative_evidence_when_applicable": true,
        "require_counter_evidence_when_applicable": true,
        "require_all_high_priority_questions_covered": true,
        "require_major_claims_supported": true,
        "require_contradictions_investigated": true,
        "maximum_search_rounds": 3,
        "maximum_total_queries": 20
    }},

    "claim_evidence_contract": {{
        "required": true,
        "require_source_location": true,
        "require_support_or_contradict_label": true,
        "distinguish_observation_interpretation_conclusion": true
    }},

    "allowed_conclusion_statuses": [
        "supported",
        "partially_supported",
        "not_supported",
        "conflicting",
        "insufficient_evidence"
    ],

    "phase_2_output_contract": {{
        "required_question_fields": [
            "id",
            "status",
            "claims",
            "quantitative_evidence",
            "limitations"
        ],

        "required_claim_fields": [
            "claim_id",
            "claim",
            "supporting_source_ids",
            "contradicting_source_ids",
            "confidence"
        ],

        "required_source_fields": [
            "source_id",
            "title",
            "url",
            "source_type",
            "publication_date",
            "evidence_eligible"
        ],

        "required_final_fields": [
            "unanswered_questions",
            "stopping_criteria",
            "overall_conclusion"
        ]
    }}
}}
"""

    # ======================================================
    # CALL GEMINI
    # ======================================================

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    if not response.text:
        raise RuntimeError(
            "Planner Agent returned an empty response."
        )

    raw_output = response.text.strip()

    # ======================================================
    # PARSE JSON
    # ======================================================

    try:
        plan = json.loads(
            raw_output
        )

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Planner Agent did not return valid JSON.\n"
            f"Received:\n{raw_output}"
        ) from error

    require_dict(
        plan,
        "planner_output"
    )

    # ======================================================
    # TOP-LEVEL VALIDATION
    # ======================================================

    required_top_fields = {
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
    }

    missing_fields = (
        required_top_fields
        - plan.keys()
    )

    if missing_fields:
        raise RuntimeError(
            "Planner output is missing: "
            + ", ".join(
                sorted(missing_fields)
            )
        )

    # ======================================================
    # DOMAIN
    # ======================================================

    domain = clean_string(
        plan["domain"]
    )

    if not domain:
        raise RuntimeError(
            "Planner returned an empty domain."
        )

    # ======================================================
    # RESEARCH MODE
    # ======================================================

    research_mode = clean_string(
        plan["research_mode"]
    ).lower()

    if (
        research_mode
        not in ALLOWED_RESEARCH_MODES
    ):
        raise RuntimeError(
            "Invalid research mode: "
            f"{research_mode}"
        )

    # ======================================================
    # TEMPORAL SENSITIVITY
    # ======================================================

    temporal_sensitivity = clean_string(
        plan["temporal_sensitivity"]
    ).lower()

    if (
        temporal_sensitivity
        not in ALLOWED_TEMPORAL_SENSITIVITY
    ):
        raise RuntimeError(
            "Invalid temporal sensitivity: "
            f"{temporal_sensitivity}"
        )

    # ======================================================
    # HYPOTHESIS
    # ======================================================

    hypothesis = plan["hypothesis"]

    if hypothesis is not None:

        if not isinstance(
            hypothesis,
            str
        ):
            raise RuntimeError(
                "Hypothesis must be text or null."
            )

        hypothesis = hypothesis.strip()

        if not hypothesis:
            hypothesis = None

    # ======================================================
    # COMPETING HYPOTHESES
    # ======================================================

    competing_hypotheses = clean_string_list(
        plan["competing_hypotheses"],
        "competing_hypotheses"
    )

    # ======================================================
    # FALSIFICATION CRITERIA
    # ======================================================

    falsification_criteria = clean_string_list(
        plan["falsification_criteria"],
        "falsification_criteria"
    )

    # ======================================================
    # EVALUATION CRITERIA
    # ======================================================

    evaluation_criteria = clean_string_list(
        plan["evaluation_criteria"],
        "evaluation_criteria"
    )

    # ======================================================
    # SOURCE POLICY
    # ======================================================

    source_policy = require_dict(
        plan["source_policy"],
        "source_policy"
    )

    required_source_policy_fields = {
        "tiers",
        "evidence_eligible_source_types",
        "discovery_only_source_types",
    }

    missing = (
        required_source_policy_fields
        - source_policy.keys()
    )

    if missing:
        raise RuntimeError(
            "source_policy is missing: "
            + ", ".join(
                sorted(missing)
            )
        )

    tiers = require_list(
        source_policy["tiers"],
        "source_policy.tiers"
    )

    if not tiers:
        raise RuntimeError(
            "source_policy.tiers cannot be empty."
        )

    cleaned_tiers = []
    seen_tiers = set()

    for index, tier in enumerate(
        tiers,
        start=1
    ):

        require_dict(
            tier,
            f"source_policy.tiers[{index}]"
        )

        required_tier_fields = {
            "tier",
            "source_types",
            "role",
        }

        missing = (
            required_tier_fields
            - tier.keys()
        )

        if missing:
            raise RuntimeError(
                f"Source tier {index} is missing: "
                + ", ".join(
                    sorted(missing)
                )
            )

        tier_number = require_positive_int(
            tier["tier"],
            f"source_policy.tiers[{index}].tier"
        )

        if tier_number in seen_tiers:
            raise RuntimeError(
                f"Duplicate source tier: "
                f"{tier_number}"
            )

        seen_tiers.add(
            tier_number
        )

        source_types = clean_string_list(
            tier["source_types"],
            (
                f"source_policy.tiers[{index}]"
                ".source_types"
            )
        )

        if not source_types:
            raise RuntimeError(
                f"Source tier {tier_number} "
                "must contain source types."
            )

        role = clean_string(
            tier["role"]
        ).lower()

        if (
            role
            not in ALLOWED_SOURCE_ROLES
        ):
            raise RuntimeError(
                f"Source tier {tier_number} "
                f"has invalid role: {role}"
            )

        cleaned_tiers.append(
            {
                "tier": tier_number,
                "source_types": source_types,
                "role": role,
            }
        )

    cleaned_tiers.sort(
        key=lambda item: item["tier"]
    )

    evidence_eligible_source_types = (
        clean_string_list(
            source_policy[
                "evidence_eligible_source_types"
            ],
            (
                "source_policy."
                "evidence_eligible_source_types"
            )
        )
    )

    discovery_only_source_types = (
        clean_string_list(
            source_policy[
                "discovery_only_source_types"
            ],
            (
                "source_policy."
                "discovery_only_source_types"
            )
        )
    )

    # ======================================================
    # GLOBAL SOURCE REQUIREMENTS
    # ======================================================

    global_requirements = require_dict(
        plan["global_source_requirements"],
        "global_source_requirements"
    )

    required_global_fields = {
        "minimum_total_sources",
        "minimum_primary_sources",
        "minimum_peer_reviewed_sources",
        "minimum_quantitative_sources",
        "minimum_counter_evidence_sources",
    }

    missing = (
        required_global_fields
        - global_requirements.keys()
    )

    if missing:
        raise RuntimeError(
            "global_source_requirements "
            "is missing: "
            + ", ".join(
                sorted(missing)
            )
        )

    cleaned_global_requirements = {}

    for field in required_global_fields:

        cleaned_global_requirements[
            field
        ] = require_non_negative_int(
            global_requirements[field],
            (
                "global_source_requirements."
                f"{field}"
            )
        )

    total_sources = (
        cleaned_global_requirements[
            "minimum_total_sources"
        ]
    )

    if total_sources <= 0:
        raise RuntimeError(
            "minimum_total_sources must be "
            "greater than zero."
        )

    for field in (
        "minimum_primary_sources",
        "minimum_peer_reviewed_sources",
        "minimum_quantitative_sources",
        "minimum_counter_evidence_sources",
    ):

        if (
            cleaned_global_requirements[
                field
            ]
            > total_sources
        ):
            raise RuntimeError(
                f"{field} cannot exceed "
                "minimum_total_sources."
            )

    # ======================================================
    # SOURCE DIVERSITY
    # ======================================================

    source_diversity_requirements = (
        clean_string_list(
            plan[
                "source_diversity_requirements"
            ],
            "source_diversity_requirements"
        )
    )

    # ======================================================
    # QUESTIONS
    # ======================================================

    questions = require_list(
        plan["questions"],
        "questions"
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

    required_question_fields = {
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

    cleaned_questions = []
    seen_ids = set()

    for index, item in enumerate(
        questions,
        start=1
    ):

        require_dict(
            item,
            f"Question {index}"
        )

        missing = (
            required_question_fields
            - item.keys()
        )

        if missing:
            raise RuntimeError(
                f"Question {index} is missing: "
                + ", ".join(
                    sorted(missing)
                )
            )

        # --------------------------------------------------
        # ID
        # --------------------------------------------------

        question_id = clean_string(
            item["id"]
        ).upper()

        if not question_id:
            raise RuntimeError(
                f"Question {index} has no ID."
            )

        if question_id in seen_ids:
            raise RuntimeError(
                "Duplicate question ID: "
                f"{question_id}"
            )

        seen_ids.add(
            question_id
        )

        # --------------------------------------------------
        # TYPE
        # --------------------------------------------------

        question_type = clean_string(
            item["type"]
        ).lower()

        if not question_type:
            raise RuntimeError(
                f"{question_id} has no type."
            )

        # --------------------------------------------------
        # PRIORITY
        # --------------------------------------------------

        priority = clean_string(
            item["priority"]
        ).lower()

        if (
            priority
            not in ALLOWED_PRIORITIES
        ):
            raise RuntimeError(
                f"{question_id} has invalid "
                f"priority: {priority}"
            )

        # --------------------------------------------------
        # QUESTION
        # --------------------------------------------------

        question_text = clean_string(
            item["question"]
        )

        if not question_text:
            raise RuntimeError(
                f"{question_id} is empty."
            )

        # --------------------------------------------------
        # EVIDENCE
        # --------------------------------------------------

        evidence_needed = clean_string_list(
            item["evidence_needed"],
            f"{question_id}.evidence_needed"
        )

        if not evidence_needed:
            raise RuntimeError(
                f"{question_id} must specify "
                "evidence requirements."
            )

        # --------------------------------------------------
        # SEARCH STRATEGY
        # --------------------------------------------------

        search_strategy = require_dict(
            item["search_strategy"],
            f"{question_id}.search_strategy"
        )

        required_search_fields = {
            "primary_queries",
            "secondary_queries",
            "counter_evidence_queries",
        }

        missing = (
            required_search_fields
            - search_strategy.keys()
        )

        if missing:
            raise RuntimeError(
                f"{question_id}.search_strategy "
                "is missing: "
                + ", ".join(
                    sorted(missing)
                )
            )

        primary_queries = clean_string_list(
            search_strategy[
                "primary_queries"
            ],
            (
                f"{question_id}.search_strategy."
                "primary_queries"
            )
        )

        secondary_queries = clean_string_list(
            search_strategy[
                "secondary_queries"
            ],
            (
                f"{question_id}.search_strategy."
                "secondary_queries"
            )
        )

        counter_queries = clean_string_list(
            search_strategy[
                "counter_evidence_queries"
            ],
            (
                f"{question_id}.search_strategy."
                "counter_evidence_queries"
            )
        )

        if not primary_queries:
            raise RuntimeError(
                f"{question_id} must have at "
                "least one primary query."
            )

        # --------------------------------------------------
        # PREFERRED SOURCE TYPES
        # --------------------------------------------------

        preferred_source_types = (
            clean_string_list(
                item[
                    "preferred_source_types"
                ],
                (
                    f"{question_id}."
                    "preferred_source_types"
                )
            )
        )

        if not preferred_source_types:
            raise RuntimeError(
                f"{question_id} must specify "
                "preferred source types."
            )

        # --------------------------------------------------
        # SOURCE REQUIREMENTS
        # --------------------------------------------------

        source_requirements = require_dict(
            item["source_requirements"],
            (
                f"{question_id}."
                "source_requirements"
            )
        )

        required_question_source_fields = {
            "minimum_sources",
            "minimum_primary_sources",
            "minimum_quantitative_sources",
            "minimum_counter_evidence_sources",
        }

        missing = (
            required_question_source_fields
            - source_requirements.keys()
        )

        if missing:
            raise RuntimeError(
                f"{question_id}.source_requirements "
                "is missing: "
                + ", ".join(
                    sorted(missing)
                )
            )

        cleaned_source_requirements = {}

        for field in (
            required_question_source_fields
        ):

            cleaned_source_requirements[
                field
            ] = require_non_negative_int(
                source_requirements[field],
                (
                    f"{question_id}."
                    "source_requirements."
                    f"{field}"
                )
            )

        minimum_sources = (
            cleaned_source_requirements[
                "minimum_sources"
            ]
        )

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

            if (
                cleaned_source_requirements[
                    field
                ]
                > minimum_sources
            ):
                raise RuntimeError(
                    f"{question_id}.{field} "
                    "cannot exceed "
                    "minimum_sources."
                )

        # --------------------------------------------------
        # QUANTITATIVE
        # --------------------------------------------------

        requires_quantitative = (
            require_bool(
                item[
                    "requires_quantitative_evidence"
                ],
                (
                    f"{question_id}."
                    "requires_quantitative_evidence"
                )
            )
        )

        quantitative_fields = (
            clean_string_list(
                item[
                    "quantitative_fields"
                ],
                (
                    f"{question_id}."
                    "quantitative_fields"
                )
            )
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

        if (
            not requires_quantitative
            and quantitative_fields
        ):
            quantitative_fields = []

        if not requires_quantitative:
                quantitative_fields = []
                cleaned_source_requirements[
                "minimum_quantitative_sources"
            ] = 0

           

        # --------------------------------------------------
        # EXPERIMENTAL CONTEXT
        # --------------------------------------------------

        experimental_context_fields = (
            clean_string_list(
                item[
                    "experimental_context_fields"
                ],
                (
                    f"{question_id}."
                    "experimental_context_fields"
                )
            )
        )

        # --------------------------------------------------
        # COUNTER EVIDENCE
        # --------------------------------------------------

        requires_counter = (
            require_bool(
                item[
                    "requires_counter_evidence"
                ],
                (
                    f"{question_id}."
                    "requires_counter_evidence"
                )
            )
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

        if not requires_counter:
                counter_queries = []
                cleaned_source_requirements[
                "minimum_counter_evidence_sources"
            ] = 0
        
            

        # --------------------------------------------------
        # BOUNDARY CONDITIONS
        # --------------------------------------------------

        boundary_conditions = (
            clean_string_list(
                item[
                    "boundary_conditions"
                ],
                (
                    f"{question_id}."
                    "boundary_conditions"
                )
            )
        )

        # --------------------------------------------------
        # DEPENDENCIES
        # --------------------------------------------------

        depends_on = [
            dependency.upper()
            for dependency
            in clean_string_list(
                item["depends_on"],
                f"{question_id}.depends_on"
            )
        ]

        # --------------------------------------------------
        # STORE QUESTION
        # --------------------------------------------------

        cleaned_questions.append(
            {
                "id":
                    question_id,

                "type":
                    question_type,

                "priority":
                    priority,

                "question":
                    question_text,

                "evidence_needed":
                    evidence_needed,

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
                    cleaned_source_requirements,

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
        )

    # ======================================================
    # REQUIRE HIGH-PRIORITY QUESTION
    # ======================================================

    if not any(
        question["priority"] == "high"
        for question
        in cleaned_questions
    ):
        raise RuntimeError(
            "Planner must generate at least one "
            "high-priority question."
        )

    # ======================================================
    # VALIDATE DEPENDENCIES
    # ======================================================

    valid_ids = {
        question["id"]
        for question
        in cleaned_questions
    }

    dependency_graph = {}

    for question in cleaned_questions:

        question_id = question["id"]

        dependency_graph[
            question_id
        ] = set(
            question["depends_on"]
        )

        for dependency in question[
            "depends_on"
        ]:

            if dependency not in valid_ids:
                raise RuntimeError(
                    f"{question_id} depends on "
                    f"unknown question "
                    f"{dependency}."
                )

            if dependency == question_id:
                raise RuntimeError(
                    f"{question_id} cannot "
                    "depend on itself."
                )

    # ======================================================
    # DETECT CIRCULAR DEPENDENCIES
    # ======================================================

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

        visiting.add(
            question_id
        )

        for dependency in (
            dependency_graph[
                question_id
            ]
        ):
            visit(
                dependency
            )

        visiting.remove(
            question_id
        )

        visited.add(
            question_id
        )

    for question_id in valid_ids:
        visit(
            question_id
        )

    # ======================================================
    # EVIDENCE SUFFICIENCY
    # ======================================================

    evidence_sufficiency = require_dict(
        plan["evidence_sufficiency"],
        "evidence_sufficiency"
    )

    required_sufficiency_fields = {
        "high",
        "medium",
        "low",
        "conflicting",
        "insufficient",
    }

    missing = (
        required_sufficiency_fields
        - evidence_sufficiency.keys()
    )

    if missing:
        raise RuntimeError(
            "evidence_sufficiency is missing: "
            + ", ".join(
                sorted(missing)
            )
        )

    cleaned_evidence_sufficiency = {}

    for field in (
        required_sufficiency_fields
    ):

        description = clean_string(
            evidence_sufficiency[field]
        )

        if not description:
            raise RuntimeError(
                "Evidence sufficiency "
                f"'{field}' cannot be empty."
            )

        cleaned_evidence_sufficiency[
            field
        ] = description

    # ======================================================
    # STOPPING CRITERIA
    # ======================================================

    stopping_criteria = require_dict(
        plan["stopping_criteria"],
        "stopping_criteria"
    )

    boolean_stopping_fields = {
        "require_minimum_total_sources",
        "require_minimum_primary_sources",
        "require_quantitative_evidence_when_applicable",
        "require_counter_evidence_when_applicable",
        "require_all_high_priority_questions_covered",
        "require_major_claims_supported",
        "require_contradictions_investigated",
    }

    required_stopping_fields = (
        boolean_stopping_fields
        | {
            "maximum_search_rounds",
            "maximum_total_queries",
        }
    )

    missing = (
        required_stopping_fields
        - stopping_criteria.keys()
    )

    if missing:
        raise RuntimeError(
            "stopping_criteria is missing: "
            + ", ".join(
                sorted(missing)
            )
        )

    cleaned_stopping_criteria = {}

    for field in (
        boolean_stopping_fields
    ):

        cleaned_stopping_criteria[
            field
        ] = require_bool(
            stopping_criteria[field],
            f"stopping_criteria.{field}"
        )

    maximum_search_rounds = (
        require_positive_int(
            stopping_criteria[
                "maximum_search_rounds"
            ],
            (
                "stopping_criteria."
                "maximum_search_rounds"
            )
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
            )
        )
    )

    cleaned_stopping_criteria[
        "maximum_search_rounds"
    ] = min(
        maximum_search_rounds,
        MAX_SEARCH_ROUNDS
    )

    cleaned_stopping_criteria[
        "maximum_total_queries"
    ] = min(
        maximum_total_queries,
        MAX_TOTAL_QUERIES
    )

    # ======================================================
    # CLAIM-EVIDENCE CONTRACT
    # ======================================================

    claim_contract = require_dict(
        plan["claim_evidence_contract"],
        "claim_evidence_contract"
    )

    required_claim_fields = {
        "required",
        "require_source_location",
        "require_support_or_contradict_label",
        "distinguish_observation_interpretation_conclusion",
    }

    missing = (
        required_claim_fields
        - claim_contract.keys()
    )

    if missing:
        raise RuntimeError(
            "claim_evidence_contract "
            "is missing: "
            + ", ".join(
                sorted(missing)
            )
        )

    cleaned_claim_contract = {}

    for field in required_claim_fields:

        cleaned_claim_contract[
            field
        ] = require_bool(
            claim_contract[field],
            (
                "claim_evidence_contract."
                f"{field}"
            )
        )

    # ======================================================
    # CONCLUSION STATUSES
    # ======================================================

    returned_statuses = {
        status.lower()
        for status
        in clean_string_list(
            plan[
                "allowed_conclusion_statuses"
            ],
            "allowed_conclusion_statuses"
        )
    }

    missing_statuses = (
        ALLOWED_CONCLUSION_STATUSES
        - returned_statuses
    )

    if missing_statuses:
        raise RuntimeError(
            "Planner is missing allowed "
            "conclusion statuses: "
            + ", ".join(
                sorted(missing_statuses)
            )
        )

    cleaned_conclusion_statuses = [
        "supported",
        "partially_supported",
        "not_supported",
        "conflicting",
        "insufficient_evidence",
    ]

    # ======================================================
    # PHASE 2 OUTPUT CONTRACT
    # ======================================================

    phase_2_contract = require_dict(
        plan["phase_2_output_contract"],
        "phase_2_output_contract"
    )

    required_phase_2_fields = {
        "required_question_fields",
        "required_claim_fields",
        "required_source_fields",
        "required_final_fields",
    }

    missing = (
        required_phase_2_fields
        - phase_2_contract.keys()
    )

    if missing:
        raise RuntimeError(
            "phase_2_output_contract "
            "is missing: "
            + ", ".join(
                sorted(missing)
            )
        )

    cleaned_phase_2_contract = {}

    for field in required_phase_2_fields:

        values = clean_string_list(
            phase_2_contract[field],
            (
                "phase_2_output_contract."
                f"{field}"
            )
        )

        if not values:
            raise RuntimeError(
                "Phase 2 output contract "
                f"'{field}' cannot be empty."
            )

        cleaned_phase_2_contract[
            field
        ] = values

    # ======================================================
    # FINAL PLAN
    # ======================================================

    return {
        "topic":
            topic,

        "domain":
            domain,

        "research_mode":
            research_mode,

        "temporal_sensitivity":
            temporal_sensitivity,

        "hypothesis":
            hypothesis,

        "competing_hypotheses":
            competing_hypotheses,

        "falsification_criteria":
            falsification_criteria,

        "evaluation_criteria":
            evaluation_criteria,

        "source_policy": {
            "tiers":
                cleaned_tiers,

            "evidence_eligible_source_types":
                evidence_eligible_source_types,

            "discovery_only_source_types":
                discovery_only_source_types,
        },

        "global_source_requirements":
            cleaned_global_requirements,

        "source_diversity_requirements":
            source_diversity_requirements,

        "questions":
            cleaned_questions,

        "evidence_sufficiency":
            cleaned_evidence_sufficiency,

        "stopping_criteria":
            cleaned_stopping_criteria,

        "claim_evidence_contract":
            cleaned_claim_contract,

        "allowed_conclusion_statuses":
            cleaned_conclusion_statuses,

        "phase_2_output_contract":
            cleaned_phase_2_contract,
    }