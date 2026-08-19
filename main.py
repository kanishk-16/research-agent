import json
import os
import sys

from dotenv import load_dotenv
from google import genai
from tavily import TavilyClient

from planner.planner import create_research_plan
# ==========================================================
# WINDOWS / TERMINAL UTF-8 SUPPORT
# ==========================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace"
    )

# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL = "gemini-3.5-flash-lite"

SEARCH_DEPTH = "advanced"

MAX_RESULTS_PER_QUERY = 5

PLAN_OUTPUT_FILE = "research_plan.json"

SEPARATOR = "=" * 70
SUB_SEPARATOR = "-" * 70


# ==========================================================
# CLIENT SETUP
# ==========================================================

def load_clients():
    """
    Load API keys and initialize Gemini and Tavily.
    """

    load_dotenv()

    gemini_key = os.getenv(
        "GEMINI_API_KEY"
    )

    tavily_key = os.getenv(
        "TAVILY_API_KEY"
    )

    if not gemini_key:
        raise ValueError(
            "GEMINI_API_KEY is missing from .env"
        )

    if not tavily_key:
        raise ValueError(
            "TAVILY_API_KEY is missing from .env"
        )

    gemini_client = genai.Client(
        api_key=gemini_key
    )

    tavily_client = TavilyClient(
        api_key=tavily_key
    )

    return (
        gemini_client,
        tavily_client
    )


# ==========================================================
# DISPLAY HELPERS
# ==========================================================

def print_header(title):
    """
    Print a major terminal heading.
    """

    print(
        "\n" + SEPARATOR
    )

    print(
        title
    )

    print(
        SEPARATOR
    )


def print_subheader(title):
    """
    Print a smaller terminal heading.
    """

    print(
        "\n" + SUB_SEPARATOR
    )

    print(
        title
    )

    print(
        SUB_SEPARATOR
    )


def print_list(
    values,
    empty_message="None"
):
    """
    Print list values as bullet points.
    """

    if not values:

        print(
            f"  {empty_message}"
        )

        return

    for value in values:

        print(
            f"  - {value}"
        )


def yes_no(value):
    """
    Convert Boolean into YES / NO.
    """

    return (
        "YES"
        if value
        else "NO"
    )


# ==========================================================
# PHASE 1
# PLANNING
# ==========================================================

def run_planner(
    gemini_client,
    topic
):
    """
    Generate the machine-executable research plan.
    """

    print_header(
        "PHASE 1 - RESEARCH PLANNING"
    )

    print(
        "\nGenerating structured research plan..."
    )

    research_plan = create_research_plan(
        gemini_client,
        MODEL,
        topic
    )

    return research_plan


# ==========================================================
# DISPLAY PLANNER OUTPUT
# ==========================================================

def display_source_policy(
    source_policy
):
    """
    Display source hierarchy.
    """

    print_subheader(
        "SOURCE QUALITY POLICY"
    )

    for tier in source_policy[
        "tiers"
    ]:

        print(
            f"\nTier {tier['tier']} "
            f"[{tier['role'].upper()}]"
        )

        print_list(
            tier[
                "source_types"
            ]
        )

    print(
        "\nEvidence-Eligible Source Types:"
    )

    print_list(
        source_policy[
            "evidence_eligible_source_types"
        ]
    )

    print(
        "\nDiscovery-Only Source Types:"
    )

    print_list(
        source_policy[
            "discovery_only_source_types"
        ]
    )


def display_global_requirements(
    requirements
):
    """
    Display global source requirements.
    """

    print_subheader(
        "GLOBAL SOURCE REQUIREMENTS"
    )

    print(
        "Minimum Total Sources: "
        f"{requirements['minimum_total_sources']}"
    )

    print(
        "Minimum Primary Sources: "
        f"{requirements['minimum_primary_sources']}"
    )

    print(
        "Minimum Peer-Reviewed Sources: "
        f"{requirements['minimum_peer_reviewed_sources']}"
    )

    print(
        "Minimum Quantitative Sources: "
        f"{requirements['minimum_quantitative_sources']}"
    )

    print(
        "Minimum Counter-Evidence Sources: "
        f"{requirements['minimum_counter_evidence_sources']}"
    )


def display_search_strategy(
    strategy
):
    """
    Display Planner-generated queries.
    """

    print(
        "\nPrimary Queries:"
    )

    print_list(
        strategy[
            "primary_queries"
        ]
    )

    print(
        "\nSecondary Queries:"
    )

    print_list(
        strategy[
            "secondary_queries"
        ]
    )

    print(
        "\nCounter-Evidence Queries:"
    )

    print_list(
        strategy[
            "counter_evidence_queries"
        ]
    )


def display_question(
    question
):
    """
    Display one research question.
    """

    print(
        "\n" + SUB_SEPARATOR
    )

    print(
        f"{question['id']} "
        f"[{question['type'].upper()}]"
    )

    print(
        "Priority: "
        f"{question['priority'].upper()}"
    )

    print(
        "\nQuestion:"
    )

    print(
        f"  {question['question']}"
    )

    print(
        "\nEvidence Needed:"
    )

    print_list(
        question[
            "evidence_needed"
        ]
    )

    display_search_strategy(
        question[
            "search_strategy"
        ]
    )

    print(
        "\nPreferred Source Types:"
    )

    print_list(
        question[
            "preferred_source_types"
        ]
    )

    requirements = question[
        "source_requirements"
    ]

    print(
        "\nQuestion Source Requirements:"
    )

    print(
        "  Minimum Sources: "
        f"{requirements['minimum_sources']}"
    )

    print(
        "  Minimum Primary Sources: "
        f"{requirements['minimum_primary_sources']}"
    )

    print(
        "  Minimum Quantitative Sources: "
        f"{requirements['minimum_quantitative_sources']}"
    )

    print(
        "  Minimum Counter-Evidence Sources: "
        f"{requirements['minimum_counter_evidence_sources']}"
    )

    print(
        "\nRequires Quantitative Evidence: "
        f"{yes_no(question['requires_quantitative_evidence'])}"
    )

    print(
        "\nQuantitative Fields:"
    )

    print_list(
        question[
            "quantitative_fields"
        ]
    )

    print(
        "\nExperimental Context:"
    )

    print_list(
        question[
            "experimental_context_fields"
        ]
    )

    print(
        "\nRequires Counter-Evidence: "
        f"{yes_no(question['requires_counter_evidence'])}"
    )

    print(
        "\nBoundary Conditions:"
    )

    print_list(
        question[
            "boundary_conditions"
        ]
    )

    print(
        "\nDepends On:"
    )

    print_list(
        question[
            "depends_on"
        ]
    )


def display_research_plan(
    research_plan
):
    """
    Display the complete Planner output.
    """

    print_header(
        "STRUCTURED RESEARCH PLAN"
    )

    print(
        f"\nTopic:\n"
        f"{research_plan['topic']}"
    )

    print(
        f"\nDomain: "
        f"{research_plan['domain']}"
    )

    print(
        "Research Mode: "
        f"{research_plan['research_mode'].upper()}"
    )

    print(
        "Temporal Sensitivity: "
        f"{research_plan['temporal_sensitivity'].upper()}"
    )

    # ------------------------------------------------------
    # Hypothesis
    # ------------------------------------------------------

    print(
        "\nWorking Hypothesis:"
    )

    hypothesis = research_plan[
        "hypothesis"
    ]

    if hypothesis:

        print(
            f"  {hypothesis}"
        )

    else:

        print(
            "  None"
        )

    # ------------------------------------------------------
    # Competing hypotheses
    # ------------------------------------------------------

    print(
        "\nCompeting Hypotheses:"
    )

    print_list(
        research_plan[
            "competing_hypotheses"
        ]
    )

    # ------------------------------------------------------
    # Falsification
    # ------------------------------------------------------

    print(
        "\nFalsification Criteria:"
    )

    print_list(
        research_plan[
            "falsification_criteria"
        ]
    )

    # ------------------------------------------------------
    # Evaluation criteria
    # ------------------------------------------------------

    print(
        "\nEvaluation / Operational Criteria:"
    )

    print_list(
        research_plan[
            "evaluation_criteria"
        ]
    )

    # ------------------------------------------------------
    # Source policy
    # ------------------------------------------------------

    display_source_policy(
        research_plan[
            "source_policy"
        ]
    )

    # ------------------------------------------------------
    # Global requirements
    # ------------------------------------------------------

    display_global_requirements(
        research_plan[
            "global_source_requirements"
        ]
    )

    # ------------------------------------------------------
    # Diversity
    # ------------------------------------------------------

    print_subheader(
        "SOURCE DIVERSITY REQUIREMENTS"
    )

    print_list(
        research_plan[
            "source_diversity_requirements"
        ]
    )

    # ------------------------------------------------------
    # Questions
    # ------------------------------------------------------

    print_header(
        "RESEARCH QUESTIONS"
    )

    for question in research_plan[
        "questions"
    ]:

        display_question(
            question
        )

    # ------------------------------------------------------
    # Evidence sufficiency
    # ------------------------------------------------------

    print_subheader(
        "EVIDENCE SUFFICIENCY"
    )

    for (
        level,
        description
    ) in research_plan[
        "evidence_sufficiency"
    ].items():

        print(
            f"\n{level.upper()}:"
        )

        print(
            f"  {description}"
        )

    # ------------------------------------------------------
    # Stopping criteria
    # ------------------------------------------------------

    criteria = research_plan[
        "stopping_criteria"
    ]

    print_subheader(
        "STOPPING CRITERIA"
    )

    print(
        "Require Minimum Total Sources: "
        f"{yes_no(criteria['require_minimum_total_sources'])}"
    )

    print(
        "Require Minimum Primary Sources: "
        f"{yes_no(criteria['require_minimum_primary_sources'])}"
    )

    print(
        "Require Quantitative Evidence: "
        f"{yes_no(criteria['require_quantitative_evidence_when_applicable'])}"
    )

    print(
        "Require Counter-Evidence: "
        f"{yes_no(criteria['require_counter_evidence_when_applicable'])}"
    )

    print(
        "Require High-Priority Coverage: "
        f"{yes_no(criteria['require_all_high_priority_questions_covered'])}"
    )

    print(
        "Require Major Claims Supported: "
        f"{yes_no(criteria['require_major_claims_supported'])}"
    )

    print(
        "Require Contradictions Investigated: "
        f"{yes_no(criteria['require_contradictions_investigated'])}"
    )

    print(
        "\nMaximum Search Rounds: "
        f"{criteria['maximum_search_rounds']}"
    )

    print(
        "Maximum Total Queries: "
        f"{criteria['maximum_total_queries']}"
    )

    print(
        "\n" + SEPARATOR
    )

    print(
        "Planner generated "
        f"{len(research_plan['questions'])} "
        "research questions."
    )

    print(
        SEPARATOR
    )


# ==========================================================
# SAVE PLAN
# ==========================================================

def save_research_plan(
    research_plan
):
    """
    Save Planner output to JSON.
    """

    with open(
        PLAN_OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            research_plan,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nResearch plan saved to "
        f"{PLAN_OUTPUT_FILE}"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    try:

        # ==================================================
        # INITIALIZE CLIENTS
        # ==================================================

        (
            gemini_client,
            tavily_client
        ) = load_clients()

        # ==================================================
        # GET TOPIC
        # ==================================================

        print_header(
            "AUTONOMOUS RESEARCH SYSTEM"
        )

        topic = input(
            "\nEnter research topic: "
        ).strip()

        if not topic:

            print(
                "\nPlease enter a research topic."
            )

            return

        # ==================================================
        # PHASE 1
        # PLANNER
        # ==================================================

        research_plan = run_planner(
            gemini_client,
            topic
        )

        # ==================================================
        # DISPLAY PLAN
        # ==================================================

        display_research_plan(
            research_plan
        )

        # ==================================================
        # SAVE PLAN
        # ==================================================

        save_research_plan(
            research_plan
        )

       

        # ==================================================
        # COMPLETE
        # ==================================================

        print_header(
            "PHASE 1 COMPLETED"
        )

        print(
            "\nResearch plan generated successfully."
        )

        print(
            f"Research plan saved to: {PLAN_OUTPUT_FILE}"
        )

        print(
            "\nPhase 2 has NOT been executed."
        )

    # ======================================================
    # KEYBOARD INTERRUPT
    # ======================================================

    except KeyboardInterrupt:

        print(
            "\n\nProgram stopped by user."
        )

        sys.exit(0)

    # ======================================================
    # ERROR HANDLING
    # ======================================================

    except Exception as error:

        print(
            f"\nError: {error}"
        )

        sys.exit(1)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()