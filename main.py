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

    if not gemini_key:
        raise ValueError(
            "GEMINI_API_KEY is missing from .env"
        )

    if not tavily_key:
        raise ValueError(
            "TAVILY_API_KEY is missing from .env"
        )

    # Gemini client
    gemini_client = genai.Client(
        api_key=gemini_key
    )

    # Tavily client
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

def search_web(tavily_client, topic):
    """
    Search the web using Tavily.

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

def build_research_context(search_results):
    """
    Convert Tavily results into text that
    can be given to Gemini.

    This is the AUGMENTATION step in our
    basic RAG-style pipeline.
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

    context = build_research_context(
        search_results
    )

    prompt = f"""
You are a careful research assistant.

Research Topic:
{topic}

Below are web search results collected from
the search engine.

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
        "Generating summary with Gemini...\n"
    )

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

def print_sources(search_results):
    """
    Display the sources retrieved by Tavily.
    """

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

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
# PLANNER AGENT
# ==========================================================


# ----------------------------------------------------------
# PHASE 2 - STEP 1
# Generate research plan
# ----------------------------------------------------------

def run_planner(
    gemini_client,
    topic
):
    """
    Run the Planner Agent and return
    a structured research plan.
    """

    print("\nPlanning research...")

    research_plan = create_research_plan(
        gemini_client,
        MODEL,
        topic
    )

    return research_plan


# ----------------------------------------------------------
# PHASE 2 - STEP 2
# Display research plan
# ----------------------------------------------------------

def display_research_plan(
    research_plan
):
    """
    Display the structured research plan.
    """

    print("\n" + "=" * 60)
    print("PHASE 2 - RESEARCH PLAN")
    print("=" * 60)

    print(
        f"\nTopic: "
        f"{research_plan['topic']}"
    )

    print(
        f"Domain: "
        f"{research_plan['domain']}"
    )

    print("\nResearch Questions:")

    for item in research_plan["questions"]:

        print("\n" + "-" * 60)

        print(
            f"{item['id']} "
            f"[{item['type'].upper()}]"
        )

        print(
            f"Priority: "
            f"{item['priority'].upper()}"
        )

        print(
            f"Question: "
            f"{item['question']}"
        )

    print("\n" + "=" * 60)

    print(
        "Planner generated "
        f"{len(research_plan['questions'])} "
        "research questions."
    )

    print("=" * 60)


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
        # PLANNING
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
        #
        # For now we still search the ORIGINAL topic.
        # Searching every Planner question belongs
        # to Phase 3.
        # ==================================================

        print("\n" + "=" * 60)
        print("PHASE 1 - BASIC RESEARCH")
        print("=" * 60)

        # ----------------------------------------------
        # Retrieval
        # ----------------------------------------------

        search_results = search_web(
            tavily_client,
            topic
        )


        # ----------------------------------------------
        # Generation
        # ----------------------------------------------

        summary = generate_summary(
            gemini_client,
            topic,
            search_results
        )


        # ----------------------------------------------
        # Display summary
        # ----------------------------------------------

        print("=" * 60)
        print("RESEARCH SUMMARY")
        print("=" * 60)

        print(summary)


        # ----------------------------------------------
        # Display sources
        # ----------------------------------------------

        print_sources(
            search_results
        )


        # ==================================================
        # FINISHED
        # ==================================================

        print("\n" + "=" * 60)
        print(
            "PHASE 1 + PHASE 2 EXECUTION COMPLETED"
        )
        print("=" * 60)


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