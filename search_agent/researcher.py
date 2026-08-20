from .query_generator import (
    fallback_search_queries,
    generate_search_queries
)
from .sources import (
    build_sources,
    deduplicate_sources
)
from .source_ranker import rank_sources


def run_research(
    tavily_client,
    topic,
    research_questions=None,
    gemini_client=None
):
    """
    Run Phase 2 research for the supplied Planner questions.
    """

    if research_questions is None:

        legacy_topic_search = True

        research_questions = [
            {
                "id": "TOPIC",
                "question": topic
            }
        ]

    else:

        legacy_topic_search = False

    if not research_questions:
        raise RuntimeError(
            "No research questions were provided."
        )

    all_results = []
    next_source_number = 1

    for question_index, question in enumerate(
        research_questions,
        start=1
    ):

        question_id = str(
            question.get(
                "id",
                f"Q{question_index}"
            )
        ).strip().upper()

        question_text = str(
            question.get(
                "question",
                ""
            )
        ).strip()

        if not question_text:
            raise RuntimeError(
                f"Research question {question_id} is empty."
            )

        if legacy_topic_search or gemini_client is None:

            query_records = fallback_search_queries(
                question
            )

            if not legacy_topic_search:

                print(
                    f"\nUsing fallback queries for {question_id}."
                )

        else:

            try:

                query_records = generate_search_queries(
                    gemini_client,
                    question
                )

            except Exception as error:

                print(
                    f"\nQuery generation fallback for {question_id}: "
                    f"{error}"
                )

                query_records = fallback_search_queries(
                    question
                )

        for query_record in query_records:

            query_text = query_record[
                "query_text"
            ]

            query_type = query_record[
                "query_type"
            ]

            print(
                f"\nSearching the web for {question_id} "
                f"[{query_type}]..."
            )

            response = tavily_client.search(
                query=query_text,
                search_depth="basic",
                max_results=3
            )

            results = response.get(
                "results",
                []
            )

            if not results:
                raise RuntimeError(
                    f"No search results were found for {question_id}."
                )

            query_record = {
                "question_id": (
                    ""
                    if legacy_topic_search
                    else question_id
                ),
                "query_text": query_text,
                "query_type": (
                    "legacy"
                    if legacy_topic_search
                    else query_type
                )
            }

            sources = build_sources(
                query_record,
                results,
                next_source_number
            )

            all_results.extend(
                sources
            )

            next_source_number += len(
                sources
            )

    candidate_count = len(
        all_results
    )

    unique_results = deduplicate_sources(
        all_results
    )

    print(
        f"\nCandidate sources: {candidate_count}"
    )

    print(
        f"Unique sources: {len(unique_results)}"
    )

    print(
        "Duplicates merged: "
        f"{candidate_count - len(unique_results)}"
    )

    return rank_sources(
        unique_results,
        research_questions
    )
