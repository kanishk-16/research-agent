from ..constants import (
    ALLOWED_CONCLUSION_STATUSES,
)

from .helpers import (
    clean_string_list,
    require_bool,
    require_dict,
)


def validate_claim_contract(value):
    """
    Validate claim-evidence traceability rules.
    """

    contract = require_dict(
        value,
        "claim_evidence_contract",
    )

    required_fields = {
        "required",
        "require_source_location",
        "require_support_or_contradict_label",
        "distinguish_observation_interpretation_conclusion",
    }

    missing = (
        required_fields
        - contract.keys()
    )

    if missing:
        raise RuntimeError(
            "claim_evidence_contract is missing: "
            + ", ".join(sorted(missing))
        )

    cleaned = {}

    for field in required_fields:
        cleaned[field] = require_bool(
            contract[field],
            (
                "claim_evidence_contract."
                f"{field}"
            ),
        )

    return cleaned


def validate_conclusion_statuses(value):
    """
    Ensure every required final conclusion
    status is available.
    """

    returned_statuses = {
        status.lower()
        for status in clean_string_list(
            value,
            "allowed_conclusion_statuses",
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

    # Return in stable canonical order.
    return [
        "supported",
        "partially_supported",
        "not_supported",
        "conflicting",
        "insufficient_evidence",
    ]


def validate_phase_2_contract(value):
    """
    Validate the output contract that Phase 2
    must eventually satisfy.
    """

    contract = require_dict(
        value,
        "phase_2_output_contract",
    )

    required_fields = {
        "required_question_fields",
        "required_claim_fields",
        "required_source_fields",
        "required_final_fields",
    }

    missing = (
        required_fields
        - contract.keys()
    )

    if missing:
        raise RuntimeError(
            "phase_2_output_contract "
            "is missing: "
            + ", ".join(sorted(missing))
        )

    cleaned = {}

    for field in required_fields:
        values = clean_string_list(
            contract[field],
            (
                "phase_2_output_contract."
                f"{field}"
            ),
        )

        if not values:
            raise RuntimeError(
                "Phase 2 output contract "
                f"'{field}' cannot be empty."
            )

        cleaned[field] = values

    return cleaned