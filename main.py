import os
import sys

from dotenv import load_dotenv
from google import genai
from tavily import TavilyClient

from agents.planner import create_research_plan


# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL = "gemini-3.5-flash-lite"


# ==========================================================
# COMMON SETUP
# Used by Phase 1 and Phase 2
# ==========================================================

def load_clients():
    """
    Load API keys from .env and create
    Gemini and Tavily clients.
    """

    load_dotenv()

    gemini_key = os.getenv("GEMINI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    # ------------------------------------------------------
    # Validate Gemini API key
    # ------------------------------------------------------

    if not gemini_key:
        raise ValueError(
            "GEMINI_API_KEY is missing from .env"
        )

    # ------------------------------------------------------
    # Validate Tavily API key
    # ------------------------------------------------------

    if not tavily_key:
        raise ValueError(
            "TAVILY_API_KEY is missing from .env"
        )

    # ------------------------------------------------------
    # Create Gemini client
    # ------------------------------------------------------

    gemini_client = genai.Client(
        api_key=gemini_key
    )

    # ------------------------------------------------------
    # Create Tavily client
    # ------------------------------------------------------

    tavily_client = TavilyClient(
        api_key=tavily_key
    )

    return gemini_client, tavily_client


# ==========================================================
# PHASE 1
# BASIC RETRIEVAL + GENERATION PIPELINE
# ==========================================================


# ----------------------------------------------------------
# PHASE 1 - STEP 1
# Search the web
# ----------------------------------------------------------

def search_web(
    tavily_client,
    topic
):
    """
    Search the original research topic
    using Tavily.

    Input:
        Research topic

    Output:
        Up to 3 relevant search results
    """

    print("\nSearching the web...")

    response = tavily_client.search(
        query=topic,
        search_depth="basic",
        max_results=3
    )

    results = response.get(
        "results",
        []
    )

    if not results:
        raise RuntimeError(
            "No search results were found."
        )

    return results


# ----------------------------------------------------------
# PHASE 1 - STEP 2
# Build research context
# ----------------------------------------------------------

def build_research_context(
    search_results
):
    """
    Convert Tavily search results into
    context that can be given to Gemini.

    This represents the AUGMENTATION step
    in the Phase 1 RAG-style pipeline.
    """

    context = ""

    for index, result in enumerate(
        search_results,
        start=1
    ):

        context += f"""
SOURCE {index}

Title:
{result.get("title", "Unknown")}

URL:
{result.get("url", "")}

Content:
{result.get("content", "")}

"""

    return context


# ----------------------------------------------------------
# PHASE 1 - STEP 3
# Generate research summary
# ----------------------------------------------------------

def generate_summary(
    gemini_client,
    topic,
    search_results
):
    """
    Give retrieved information to Gemini
    and generate a readable research summary.
    """

    # ------------------------------------------------------
    # Build augmented context
    # ------------------------------------------------------

    context = build_research_context(
        search_results
    )

    # ------------------------------------------------------
    # Construct generation prompt
    # ------------------------------------------------------

    prompt = f"""
You are a careful research assistant.

Research Topic:

{topic}

Below are web search results collected
from the search engine.

{context}

Write a concise research summary based ONLY
on the information provided above.

Requirements:

1. Do not invent facts.

2. Do not use unsupported information.

3. Mention important dates and numbers
   when relevant.

4. If sources disagree, mention the
   disagreement.

5. Cite information using:
   [Source 1], [Source 2], [Source 3], etc.

6. Write approximately 2-4 paragraphs.

7. Do not create fake URLs.
"""

    print(
        "\nGenerating summary with Gemini...\n"
    )

    # ------------------------------------------------------
    # Send augmented prompt to Gemini
    # ------------------------------------------------------

    response = (
        gemini_client.models.generate_content(
            model=MODEL,
            contents=prompt
        )
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return response.text


# ----------------------------------------------------------
# PHASE 1 - STEP 4
# Display sources
# ----------------------------------------------------------

def print_sources(
    search_results
):
    """
    Display the sources retrieved by Tavily.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "SOURCES"
    )

    print(
        "=" * 70
    )

    for index, result in enumerate(
        search_results,
        start=1
    ):

        title = result.get(
            "title",
            "Unknown"
        )

        url = result.get(
            "url",
            ""
        )

        print(
            f"\n[Source {index}] {title}"
        )

        print(url)


# ==========================================================
# PHASE 2
# INTELLIGENT RESEARCH PLANNER
# ==========================================================


# ----------------------------------------------------------
# PHASE 2 - STEP 1
# Run Planner Agent
# ----------------------------------------------------------

def run_planner(
    gemini_client,
    topic
):
    """
    Run the Planner Agent.

    The Planner creates a structured
    research investigation plan.
    """

    print(
        "\nPlanning research..."
    )

    research_plan = create_research_plan(
        gemini_client,
        MODEL,
        topic
    )

    return research_plan


# ----------------------------------------------------------
# PHASE 2 - STEP 2
# Helper function for lists
# ----------------------------------------------------------

def print_list(
    values
):
    """
    Display values as bullet points.
    """

    if not values:

        print(
            "  None"
        )

        return

    for value in values:

        print(
            f"  - {value}"
        )


# ----------------------------------------------------------
# PHASE 2 - STEP 3
# Display complete research plan
# ----------------------------------------------------------

def display_research_plan(
    research_plan
):
    """
    Display the structured research
    investigation plan generated by
    the Planner Agent.
    """

    # ======================================================
    # HEADER
    # ======================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 2 - RESEARCH INVESTIGATION PLAN"
    )

    print(
        "=" * 70
    )


    # ======================================================
    # TOPIC
    # ======================================================

    print(
        f"\nTopic:\n"
        f"{research_plan['topic']}"
    )


    # ======================================================
    # DOMAIN
    # ======================================================

    print(
        f"\nDomain: "
        f"{research_plan['domain']}"
    )


    # ======================================================
    # RESEARCH MODE
    # ======================================================

    print(
        f"Research Mode: "
        f"{research_plan['research_mode'].upper()}"
    )


    # ======================================================
    # WORKING HYPOTHESIS
    # ======================================================

    hypothesis = research_plan[
        "hypothesis"
    ]

    print(
        "\nWorking Hypothesis:"
    )

    if hypothesis:

        print(
            f"  {hypothesis}"
        )

    else:

        print(
            "  None - a hypothesis is not "
            "required for this exploratory "
            "or descriptive investigation."
        )


    # ======================================================
    # EVALUATION / OPERATIONAL CRITERIA
    # ======================================================

    print(
        "\nEvaluation / Operational Criteria:"
    )

    evaluation_criteria = research_plan[
        "evaluation_criteria"
    ]

    if evaluation_criteria:

        print_list(
            evaluation_criteria
        )

    else:

        print(
            "  None required for this topic."
        )


    # ======================================================
    # RESEARCH QUESTIONS
    # ======================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RESEARCH QUESTIONS"
    )

    print(
        "=" * 70
    )


    # ======================================================
    # DISPLAY EACH RESEARCH QUESTION
    # ======================================================

    for item in research_plan[
        "questions"
    ]:

        print(
            "\n" + "-" * 70
        )


        # --------------------------------------------------
        # Question ID and type
        # --------------------------------------------------

        print(
            f"{item['id']} "
            f"[{item['type'].upper()}]"
        )


        # --------------------------------------------------
        # Priority
        # --------------------------------------------------

        print(
            f"Priority: "
            f"{item['priority'].upper()}"
        )


        # --------------------------------------------------
        # Research question
        # --------------------------------------------------

        print(
            "\nQuestion:"
        )

        print(
            f"  {item['question']}"
        )


        # --------------------------------------------------
        # Evidence requirements
        # --------------------------------------------------

        print(
            "\nEvidence Needed:"
        )

        print_list(
            item[
                "evidence_needed"
            ]
        )


        # --------------------------------------------------
        # Preferred source types
        # --------------------------------------------------

        print(
            "\nPreferred Source Types:"
        )

        print_list(
            item[
                "preferred_source_types"
            ]
        )


        # --------------------------------------------------
        # Quantitative evidence requirement
        # --------------------------------------------------

        quantitative = (
            "YES"
            if item[
                "requires_quantitative_evidence"
            ]
            else "NO"
        )

        print(
            "\nRequires Quantitative Evidence: "
            f"{quantitative}"
        )


        # --------------------------------------------------
        # Counter-evidence requirement
        # --------------------------------------------------

        counter_evidence = (
            "YES"
            if item[
                "requires_counter_evidence"
            ]
            else "NO"
        )

        print(
            "Requires Counter-Evidence Search: "
            f"{counter_evidence}"
        )


        # --------------------------------------------------
        # Boundary conditions
        # --------------------------------------------------

        print(
            "\nBoundary Conditions:"
        )

        print_list(
            item[
                "boundary_conditions"
            ]
        )


        # --------------------------------------------------
        # Question dependencies
        # --------------------------------------------------

        print(
            "\nDepends On:"
        )

        print_list(
            item[
                "depends_on"
            ]
        )


    # ======================================================
    # PLANNER SUMMARY
    # ======================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "Planner generated "
        f"{len(research_plan['questions'])} "
        "research questions."
    )

    print(
        "=" * 70
    )


# ==========================================================
# MAIN PROGRAM
# ==========================================================

def main():

    try:

        # ==================================================
        # INITIALIZATION
        # ==================================================

        gemini_client, tavily_client = (
            load_clients()
        )


        # ==================================================
        # GET USER TOPIC
        # ==================================================

        topic = input(
            "Enter research topic: "
        ).strip()

        if not topic:

            print(
                "Please enter a research topic."
            )

            return


        # ==================================================
        # PHASE 2
        # RESEARCH PLANNING
        # ==================================================

        research_plan = run_planner(
            gemini_client,
            topic
        )

        display_research_plan(
            research_plan
        )


        # ==================================================
        # PHASE 1
        # BASIC RESEARCH PIPELINE
        # ==================================================
        #
        # IMPORTANT:
        #
        # Phase 2 has now created a detailed research plan.
        #
        # However, the current Phase 1 pipeline still searches
        # ONLY the original topic.
        #
        # It does NOT yet:
        #
        # - search each Planner question
        # - use evidence requirements
        # - enforce preferred source types
        # - perform counter-evidence searches
        # - use question dependencies
        #
        # Those features belong to Phase 3.
        #
        # ==================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "PHASE 1 - BASIC RESEARCH PIPELINE"
        )

        print(
            "=" * 70
        )


        # --------------------------------------------------
        # PHASE 1 - Retrieval
        # --------------------------------------------------

        search_results = search_web(
            tavily_client,
            topic
        )


        # --------------------------------------------------
        # PHASE 1 - Generation
        # --------------------------------------------------

        summary = generate_summary(
            gemini_client,
            topic,
            search_results
        )


        # --------------------------------------------------
        # Display Phase 1 summary
        # --------------------------------------------------

        print(
            "=" * 70
        )

        print(
            "RESEARCH SUMMARY"
        )

        print(
            "=" * 70
        )

        print(
            summary
        )


        # --------------------------------------------------
        # Display Phase 1 sources
        # --------------------------------------------------

        print_sources(
            search_results
        )


        # ==================================================
        # EXECUTION FINISHED
        # ==================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "PHASE 1 + PHASE 2 EXECUTION COMPLETED"
        )

        print(
            "=" * 70
        )


    # ======================================================
    # USER INTERRUPT
    # ======================================================

    except KeyboardInterrupt:

        print(
            "\n\nProgram stopped by user."
        )

        sys.exit(0)


    # ======================================================
    # GENERAL ERROR HANDLING
    # ======================================================

    except Exception as error:

        print(
            f"\nError: {error}"
        )

        sys.exit(1)


# ==========================================================
# PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()