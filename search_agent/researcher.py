from .query_generator import (
    fallback_search_queries,
    generate_search_queries
)
from .sources import (
    build_sources,
    deduplicate_sources
)
from .source_ranker import rank_sources
from .source_selector import select_sources
from .content_retriever import retrieve_selected_sources


MAX_RESULTS_PER_QUERY = 5


class ResearchPackage(list):
    """
    Subclass of list containing canonical ranked sources and carrying
    Phase 2 selection, provenance, and retrieval metadata attributes.
    """
    def __init__(
        self,
        sources,
        selection_bundle=None,
        unique_selected_sources=None,
        retrieval_stats=None,
        duplicates_merged=0
    ):
        super().__init__(sources)
        self.selection_bundle = selection_bundle or {}
        self.unique_selected_sources = unique_selected_sources or []
        self.retrieval_stats = retrieval_stats or {}
        self.duplicates_merged = duplicates_merged


def run_research(
    tavily_client,
    topic,
    research_questions=None,
    gemini_client=None,
    research_plan=None
):
    """
    Run Phase 2 research for the supplied Planner questions:
    Discovery -> Deduplication -> Ranking -> Per-Question Selection -> Deep Content Retrieval.
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

    stopping_criteria = {}
    if research_plan and isinstance(research_plan, dict):
        stopping_criteria = research_plan.get("stopping_criteria", {}) or {}

    max_total_queries = stopping_criteria.get("maximum_total_queries")
    max_search_rounds = stopping_criteria.get("maximum_search_rounds")

    total_queries_executed = 0

    all_results = []
    next_source_number = 1

    for question_index, question in enumerate(
        research_questions,
        start=1
    ):

        current_round = 0

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

            if max_search_rounds is not None and current_round >= max_search_rounds:
                print(
                    f"\nSkipping remaining queries for {question_id}: "
                    f"maximum search rounds ({max_search_rounds}) reached."
                )
                break

            if max_total_queries is not None and total_queries_executed >= max_total_queries:
                print(
                    f"\nStopping search: "
                    f"maximum total queries ({max_total_queries}) reached."
                )
                break

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

            import time

            response = None
            for search_attempt in range(3):
                try:
                    response = tavily_client.search(
                        query=query_text,
                        search_depth="basic",
                        max_results=MAX_RESULTS_PER_QUERY
                    )
                    break
                except Exception as search_err:
                    print(
                        f"  [Warning] Tavily search attempt {search_attempt + 1} failed: {search_err}"
                    )
                    if search_attempt < 2:
                        time.sleep(3 * (search_attempt + 1))
                    else:
                        response = None

            total_queries_executed += 1
            current_round += 1

            if not response or not isinstance(response, dict):
                print(
                    f"  [Warning] Skipping query '{query_text}' due to persistent API error."
                )
                continue

            results = response.get(
                "results",
                []
            )

            if not results:
                print(
                    f"  [Warning] No search results returned for '{query_text}'."
                )
                continue

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

        if max_total_queries is not None and total_queries_executed >= max_total_queries:
            break

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

    # 1. Rank canonical sources per question
    ranked_sources = rank_sources(
        unique_results,
        research_questions
    )

    # 2. Per-question source selection with counter-evidence protection
    selection_bundle = select_sources(
        research_questions,
        ranked_sources
    )

    unique_selected = selection_bundle["unique_selected_sources"]

    print(
        f"\nSelected unique sources for retrieval: {len(unique_selected)}"
    )

    # 3. Deep content retrieval for unique selected sources
    retrieval_stats = retrieve_selected_sources(
        tavily_client,
        unique_selected
    )

    print(
        f"Deep content retrieval completed: "
        f"{retrieval_stats['successes']} full, "
        f"{retrieval_stats['partials']} partial, "
        f"{retrieval_stats['snippet_only']} snippet-only, "
        f"{retrieval_stats['failures']} failed."
    )

    return ResearchPackage(
        ranked_sources,
        selection_bundle=selection_bundle,
        unique_selected_sources=unique_selected,
        retrieval_stats=retrieval_stats,
        duplicates_merged=candidate_count - len(unique_results)
    )
