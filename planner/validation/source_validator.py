from ..constants import ALLOWED_SOURCE_ROLES

from .helpers import (
    clean_string,
    clean_string_list,
    require_dict,
    require_list,
    require_non_negative_int,
    require_positive_int,
)


def validate_source_policy(value):
    """
    Validate and normalize the Planner source policy.
    """

    source_policy = require_dict(
        value,
        "source_policy",
    )

    required_fields = {
        "tiers",
        "evidence_eligible_source_types",
        "discovery_only_source_types",
    }

    missing = required_fields - source_policy.keys()

    if missing:
        raise RuntimeError(
            "source_policy is missing: "
            + ", ".join(sorted(missing))
        )

    tiers = require_list(
        source_policy["tiers"],
        "source_policy.tiers",
    )

    if not tiers:
        raise RuntimeError(
            "source_policy.tiers cannot be empty."
        )

    cleaned_tiers = []
    seen_tiers = set()

    for index, tier in enumerate(tiers, start=1):
        require_dict(
            tier,
            f"source_policy.tiers[{index}]",
        )

        required_tier_fields = {
            "tier",
            "source_types",
            "role",
        }

        missing = required_tier_fields - tier.keys()

        if missing:
            raise RuntimeError(
                f"Source tier {index} is missing: "
                + ", ".join(sorted(missing))
            )

        tier_number = require_positive_int(
            tier["tier"],
            f"source_policy.tiers[{index}].tier",
        )

        if tier_number in seen_tiers:
            raise RuntimeError(
                f"Duplicate source tier: {tier_number}"
            )

        seen_tiers.add(tier_number)

        source_types = clean_string_list(
            tier["source_types"],
            f"source_policy.tiers[{index}].source_types",
        )

        if not source_types:
            raise RuntimeError(
                f"Source tier {tier_number} "
                "must contain source types."
            )

        role = clean_string(
            tier["role"]
        ).lower()

        if role not in ALLOWED_SOURCE_ROLES:
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

    evidence_eligible_source_types = clean_string_list(
        source_policy[
            "evidence_eligible_source_types"
        ],
        "source_policy.evidence_eligible_source_types",
    )

    discovery_only_source_types = clean_string_list(
        source_policy[
            "discovery_only_source_types"
        ],
        "source_policy.discovery_only_source_types",
    )

    return {
        "tiers": cleaned_tiers,
        "evidence_eligible_source_types":
            evidence_eligible_source_types,
        "discovery_only_source_types":
            discovery_only_source_types,
    }


def validate_global_source_requirements(value):
    """
    Validate global minimum source requirements.
    """

    requirements = require_dict(
        value,
        "global_source_requirements",
    )

    required_fields = {
        "minimum_total_sources",
        "minimum_primary_sources",
        "minimum_peer_reviewed_sources",
        "minimum_quantitative_sources",
        "minimum_counter_evidence_sources",
    }

    missing = required_fields - requirements.keys()

    if missing:
        raise RuntimeError(
            "global_source_requirements is missing: "
            + ", ".join(sorted(missing))
        )

    cleaned = {}

    for field in required_fields:
        cleaned[field] = require_non_negative_int(
            requirements[field],
            f"global_source_requirements.{field}",
        )

    total_sources = cleaned[
        "minimum_total_sources"
    ]

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
        if cleaned[field] > total_sources:
            raise RuntimeError(
                f"{field} cannot exceed "
                "minimum_total_sources."
            )

    return cleaned