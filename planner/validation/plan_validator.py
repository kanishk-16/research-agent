from ..constants import (
    ALLOWED_RESEARCH_MODES,
    ALLOWED_TEMPORAL_SENSITIVITY,
)

from .helpers import (
    clean_string,
    clean_string_list,
    require_dict,
)

from .source_validator import (
    validate_global_source_requirements,
    validate_source_policy,
)

from .question_validator import (
    validate_questions,
)

from .stopping_validator import (
    validate_evidence_sufficiency,
    validate_stopping_criteria,
)

from .contract_validator import (
    validate_claim_contract,
    validate_conclusion_statuses,
    validate_phase_2_contract,
)


REQUIRED_TOP_FIELDS = {
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


def validate_plan(plan, topic):
    """
    Validate and normalize the complete
    Planner Agent response.
    """

    require_dict(
        plan,
        "planner_output",
    )

    missing_fields = (
        REQUIRED_TOP_FIELDS
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
        if not isinstance(hypothesis, str):
            raise RuntimeError(
                "Hypothesis must be text or null."
            )

        hypothesis = hypothesis.strip()

        if not hypothesis:
            hypothesis = None

    # ======================================================
    # COMPETING HYPOTHESES
    # ======================================================

    competing_hypotheses = (
        clean_string_list(
            plan["competing_hypotheses"],
            "competing_hypotheses",
        )
    )

    # ======================================================
    # FALSIFICATION CRITERIA
    # ======================================================

    falsification_criteria = (
        clean_string_list(
            plan["falsification_criteria"],
            "falsification_criteria",
        )
    )

    # ======================================================
    # EVALUATION CRITERIA
    # ======================================================

    evaluation_criteria = (
        clean_string_list(
            plan["evaluation_criteria"],
            "evaluation_criteria",
        )
    )

    # ======================================================
    # SOURCE POLICY
    # ======================================================

    source_policy = validate_source_policy(
        plan["source_policy"]
    )

    # ======================================================
    # GLOBAL SOURCE REQUIREMENTS
    # ======================================================

    global_source_requirements = (
        validate_global_source_requirements(
            plan[
                "global_source_requirements"
            ]
        )
    )

    # ======================================================
    # SOURCE DIVERSITY
    # ======================================================

    source_diversity_requirements = (
        clean_string_list(
            plan[
                "source_diversity_requirements"
            ],
            "source_diversity_requirements",
        )
    )

    # ======================================================
    # QUESTIONS
    # ======================================================

    questions = validate_questions(
        plan["questions"]
    )

    # ======================================================
    # EVIDENCE SUFFICIENCY
    # ======================================================

    evidence_sufficiency = (
        validate_evidence_sufficiency(
            plan["evidence_sufficiency"]
        )
    )

    # ======================================================
    # STOPPING CRITERIA
    # ======================================================

    stopping_criteria = (
        validate_stopping_criteria(
            plan["stopping_criteria"]
        )
    )

    # ======================================================
    # CLAIM-EVIDENCE CONTRACT
    # ======================================================

    claim_evidence_contract = (
        validate_claim_contract(
            plan["claim_evidence_contract"]
        )
    )

    # ======================================================
    # CONCLUSION STATUSES
    # ======================================================

    allowed_conclusion_statuses = (
        validate_conclusion_statuses(
            plan[
                "allowed_conclusion_statuses"
            ]
        )
    )

    # ======================================================
    # PHASE 2 OUTPUT CONTRACT
    # ======================================================

    phase_2_output_contract = (
        validate_phase_2_contract(
            plan["phase_2_output_contract"]
        )
    )

    # ======================================================
    # FINAL NORMALIZED PLAN
    # ======================================================

    return {
        "topic": topic,

        "domain": domain,

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

        "source_policy":
            source_policy,

        "global_source_requirements":
            global_source_requirements,

        "source_diversity_requirements":
            source_diversity_requirements,

        "questions":
            questions,

        "evidence_sufficiency":
            evidence_sufficiency,

        "stopping_criteria":
            stopping_criteria,

        "claim_evidence_contract":
            claim_evidence_contract,

        "allowed_conclusion_statuses":
            allowed_conclusion_statuses,

        "phase_2_output_contract":
            phase_2_output_contract,
    }