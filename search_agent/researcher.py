import asyncio
import random
from concurrent.futures import ThreadPoolExecutor

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
from .source_providers import (
    OpenAlexProvider,
    SemanticScholarProvider,
)


MAX_RESULTS_PER_QUERY = 5
OPENALEX_ENABLED = True
SEMANTIC_SCHOLAR_ENABLED = True


_ACADEMIC_ROUTE_KEYWORDS = {
    "paper", "preprint", "academic", "peer-reviewed",
    "review", "study", "government", "official",
    "standard", "documentation", "journal", "conference",
}


def _should_route_to_openalex(question):
    preferred = question.get("preferred_source_types", [])
    reqs = question.get("source_requirements", {})

    if reqs.get("minimum_primary_sources", 0) > 0:
        return True

    for p in preferred:
        p_lower = str(p).lower()
        if any(kw in p_lower for kw in _ACADEMIC_ROUTE_KEYWORDS):
            return True

    return False


def _compute_backoff_delay(attempt: int, base_delay: float = 0.5, max_delay: float = 8.0) -> float:
    """Compute exponential backoff delay with full jitter."""
    exp_cap = min(max_delay, base_delay * (2 ** attempt))
    return round(random.uniform(base_delay * 0.5, exp_cap), 3)


def _run_coro_sync(coro):
    """Run an async coroutine from synchronous code, even if an event loop is already active."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


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


async def _query_tavily_async(tavily_client, query_text: str):
    """Query Tavily asynchronously with non-blocking exponential backoff with jitter."""
    if tavily_client is None:
        return None
    for search_attempt in range(3):
        try:
            return await asyncio.to_thread(
                tavily_client.search,
                query=query_text,
                search_depth="basic",
                max_results=MAX_RESULTS_PER_QUERY,
            )
        except Exception as search_err:
            print(f"  [Warning] Tavily search attempt {search_attempt + 1} failed: {search_err}")
            if search_attempt < 2:
                await asyncio.sleep(_compute_backoff_delay(search_attempt))
    return None


async def _query_openalex_async(openalex_provider, query_text: str, question_id: str, enabled: bool):
    """Query OpenAlex asynchronously in a worker thread."""
    if not enabled or openalex_provider is None:
        return []
    try:
        res = await asyncio.to_thread(openalex_provider.search, query_text, question_id=question_id)
        if res:
            print(f"  OpenAlex: {len(res)} academic candidates for {question_id}")
        return res or []
    except Exception as exc:
        print(f"  [Warning] OpenAlex failed for {question_id}: {exc}")
        return []


async def _query_semantic_scholar_async(semantic_scholar_provider, query_text: str, question_id: str, enabled: bool):
    """Query Semantic Scholar asynchronously in a worker thread."""
    if not enabled or semantic_scholar_provider is None:
        return []
    try:
        res = await asyncio.to_thread(semantic_scholar_provider.search, query_text, question_id=question_id)
        if res:
            print(f"  Semantic Scholar: {len(res)} academic candidates for {question_id}")
        return res or []
    except Exception as exc:
        print(f"  [Warning] Semantic Scholar failed for {question_id}: {exc}")
        return []


async def _execute_single_query_bundle_async(
    task_spec: dict,
    tavily_client,
    openalex_provider,
    semantic_scholar_provider,
    semaphore: asyncio.Semaphore,
):
    """
    Execute Tavily, OpenAlex, and Semantic Scholar concurrently for one query record
    using asyncio.gather().
    """
    async with semaphore:
        question_id = task_spec["question_id"]
        query_text = task_spec["query_text"]
        query_type = task_spec["query_type"]
        route_academic = task_spec["route_academic"]
        legacy_topic_search = task_spec["legacy_topic_search"]

        print(f"\nSearching concurrently for {question_id} [{query_type}]: {query_text[:80]}...")

        use_oa = OPENALEX_ENABLED and openalex_provider is not None and query_type == "normal" and not legacy_topic_search and route_academic
        use_s2 = SEMANTIC_SCHOLAR_ENABLED and semantic_scholar_provider is not None and query_type == "normal" and not legacy_topic_search and route_academic

        tavily_resp, oa_sources, s2_sources = await asyncio.gather(
            _query_tavily_async(tavily_client, query_text),
            _query_openalex_async(openalex_provider, query_text, question_id, use_oa),
            _query_semantic_scholar_async(semantic_scholar_provider, query_text, question_id, use_s2),
        )

        tavily_results = []
        if isinstance(tavily_resp, dict):
            tavily_results = tavily_resp.get("results", []) or []
        else:
            print(f"  [Warning] Skipping query '{query_text}' due to persistent API error.")

        return {
            "task_spec": task_spec,
            "tavily_results": tavily_results,
            "oa_sources": oa_sources,
            "s2_sources": s2_sources,
        }


async def _resolve_question_queries_async(
    question_index: int,
    question: dict,
    legacy_topic_search: bool,
    gemini_client,
    research_plan,
):
    """Resolve query records for a single question (from plan, LLM generator, or fallback)."""
    question_id = str(question.get("id", f"Q{question_index}")).strip().upper()
    question_text = str(question.get("question", "")).strip()
    if not question_text:
        raise RuntimeError(f"Research question {question_id} is empty.")

    strategy = question.get("search_strategy") if isinstance(question.get("search_strategy"), dict) else None
    plan_primary = (strategy.get("primary_queries") or []) if strategy else []
    plan_counter = (strategy.get("counter_evidence_queries") or []) if strategy else []

    if legacy_topic_search or gemini_client is None:
        query_records = fallback_search_queries(question)
    elif research_plan is not None and plan_primary:
        query_records = [
            {
                "question_id": question_id,
                "query_text": str(q).strip(),
                "query_type": "normal",
            }
            for q in plan_primary[:2]
            if str(q).strip()
        ]
        if question.get("requires_counter_evidence", False) and plan_counter:
            for cq in plan_counter[:1]:
                if str(cq).strip():
                    query_records.append({
                        "question_id": question_id,
                        "query_text": str(cq).strip(),
                        "query_type": "counter",
                    })
    else:
        try:
            query_records = await asyncio.to_thread(generate_search_queries, gemini_client, question)
        except Exception as error:
            print(f"\nQuery generation fallback for {question_id}: {error}")
            query_records = fallback_search_queries(question)

    return question_id, question, query_records


async def run_research_async(
    tavily_client,
    topic,
    research_questions=None,
    gemini_client=None,
    research_plan=None,
    openalex_provider=None,
    semantic_scholar_provider=None,
    max_concurrency: int = 8,
):
    """
    Asynchronous Phase 2 research pipeline using asyncio.gather():
    Concurrently queries Tavily, OpenAlex, and Semantic Scholar across all sub-questions,
    then deduplicates, ranks, selects, and retrieves deep full-text content in parallel.
    """
    if semantic_scholar_provider is None and SEMANTIC_SCHOLAR_ENABLED:
        semantic_scholar_provider = SemanticScholarProvider(tavily_client=tavily_client)

    if research_questions is None:
        legacy_topic_search = True
        research_questions = [{"id": "TOPIC", "question": topic}]
    else:
        legacy_topic_search = False

    if not research_questions:
        raise RuntimeError("No research questions were provided.")

    stopping_criteria = {}
    if research_plan and isinstance(research_plan, dict):
        stopping_criteria = research_plan.get("stopping_criteria", {}) or {}

    max_total_queries = stopping_criteria.get("maximum_total_queries")
    max_search_rounds = stopping_criteria.get("maximum_search_rounds")

    # 1. Resolve query lists for all sub-questions concurrently
    resolved_questions = await asyncio.gather(
        *[
            _resolve_question_queries_async(idx, q, legacy_topic_search, gemini_client, research_plan)
            for idx, q in enumerate(research_questions, start=1)
        ]
    )

    # 2. Build flat list of query tasks respecting per-question and total query budgets
    scheduled_specs = []
    total_queries_executed = 0

    for question_id, question, query_records in resolved_questions:
        current_round = 0
        route_academic = _should_route_to_openalex(question)
        for qrec in query_records:
            if max_search_rounds is not None and current_round >= max_search_rounds:
                break
            if max_total_queries is not None and total_queries_executed >= max_total_queries:
                break
            q_text = str(qrec.get("query_text", "")).strip()
            if not q_text:
                continue
            scheduled_specs.append({
                "question_id": question_id,
                "query_text": q_text,
                "query_type": str(qrec.get("query_type", "normal")).strip().lower(),
                "route_academic": route_academic,
                "legacy_topic_search": legacy_topic_search,
            })
            current_round += 1
            total_queries_executed += 1
        if max_total_queries is not None and total_queries_executed >= max_total_queries:
            break

    # 3. Execute all queries across Tavily, OpenAlex, and Semantic Scholar concurrently via asyncio.gather()
    semaphore = asyncio.Semaphore(max_concurrency)
    bundle_outputs = await asyncio.gather(
        *[
            _execute_single_query_bundle_async(
                spec,
                tavily_client,
                openalex_provider,
                semantic_scholar_provider,
                semaphore,
            )
            for spec in scheduled_specs
        ]
    )

    # 4. Assemble results in deterministic scheduled order
    all_results = []
    next_source_number = 1

    for out in bundle_outputs:
        spec = out["task_spec"]
        tavily_results = out["tavily_results"]
        if tavily_results:
            q_rec_out = {
                "question_id": "" if spec["legacy_topic_search"] else spec["question_id"],
                "query_text": spec["query_text"],
                "query_type": "legacy" if spec["legacy_topic_search"] else spec["query_type"],
            }
            sources = build_sources(q_rec_out, tavily_results, next_source_number)
            all_results.extend(sources)
            next_source_number += len(sources)

        if out["oa_sources"]:
            all_results.extend(out["oa_sources"])
        if out["s2_sources"]:
            all_results.extend(out["s2_sources"])

    candidate_count = len(all_results)
    unique_results = deduplicate_sources(all_results)

    print(f"\nCandidate sources: {candidate_count}")
    print(f"Unique sources: {len(unique_results)}")
    print(f"Duplicates merged: {candidate_count - len(unique_results)}")

    source_policy = research_plan.get("source_policy") if isinstance(research_plan, dict) else None
    ranked_sources = rank_sources(
        unique_results,
        research_questions,
        source_policy=source_policy
    )

    selection_bundle = select_sources(
        research_questions,
        ranked_sources
    )
    unique_selected = selection_bundle["unique_selected_sources"]
    print(f"\nSelected unique sources for retrieval: {len(unique_selected)}")

    retrieval_stats = await asyncio.to_thread(
        retrieve_selected_sources,
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


def run_research(
    tavily_client,
    topic,
    research_questions=None,
    gemini_client=None,
    research_plan=None,
    openalex_provider=None,
    semantic_scholar_provider=None
):
    """
    Synchronous entrypoint wrapping the concurrent asyncio.gather() search pipeline.
    """
    return _run_coro_sync(
        run_research_async(
            tavily_client=tavily_client,
            topic=topic,
            research_questions=research_questions,
            gemini_client=gemini_client,
            research_plan=research_plan,
            openalex_provider=openalex_provider,
            semantic_scholar_provider=semantic_scholar_provider,
        )
    )


async def run_targeted_research_async(
    tavily_client,
    topic,
    targeted_questions,
    gemini_client=None,
    research_plan=None,
    openalex_provider=None,
    semantic_scholar_provider=None,
    max_concurrency: int = 8,
):
    """
    Asynchronous focused research pass using asyncio.gather() across all targeted queries and providers.
    """
    if semantic_scholar_provider is None and SEMANTIC_SCHOLAR_ENABLED:
        semantic_scholar_provider = SemanticScholarProvider(tavily_client=tavily_client)

    if not targeted_questions:
        return ResearchPackage([])

    research_questions = []
    query_map = {}

    for entry in targeted_questions:
        question = entry.get("question", {})
        qid = str(question.get("id", "")).strip().upper()
        research_questions.append(question)
        query_map[qid] = entry.get("targeted_queries", [])

    stopping_criteria = {}
    if research_plan and isinstance(research_plan, dict):
        stopping_criteria = research_plan.get("stopping_criteria", {}) or {}

    max_total_queries = stopping_criteria.get("maximum_total_queries")
    max_search_rounds = stopping_criteria.get("maximum_search_rounds")

    scheduled_specs = []
    total_queries_executed = 0

    for question_index, question in enumerate(research_questions, start=1):
        current_round = 0
        question_id = str(question.get("id", f"Q{question_index}")).strip().upper()
        targeted_queries = query_map.get(question_id, [])
        if not targeted_queries:
            continue

        route_academic = _should_route_to_openalex(question)
        for query_record in targeted_queries:
            if max_search_rounds is not None and current_round >= max_search_rounds:
                break
            if max_total_queries is not None and total_queries_executed >= max_total_queries:
                break
            query_text = str(query_record.get("query_text", "")).strip()
            query_type = str(query_record.get("query_type", "normal")).strip().lower()
            if not query_text:
                continue

            scheduled_specs.append({
                "question_id": question_id,
                "query_text": query_text,
                "query_type": query_type,
                "route_academic": route_academic,
                "legacy_topic_search": False,
            })
            total_queries_executed += 1
            current_round += 1

        if max_total_queries is not None and total_queries_executed >= max_total_queries:
            break

    if not scheduled_specs:
        return ResearchPackage([])

    semaphore = asyncio.Semaphore(max_concurrency)
    bundle_outputs = await asyncio.gather(
        *[
            _execute_single_query_bundle_async(
                spec,
                tavily_client,
                openalex_provider,
                semantic_scholar_provider,
                semaphore,
            )
            for spec in scheduled_specs
        ]
    )

    all_results = []
    next_source_number = 1

    for out in bundle_outputs:
        spec = out["task_spec"]
        tavily_results = out["tavily_results"]
        if tavily_results:
            q_rec_out = {
                "question_id": spec["question_id"],
                "query_text": spec["query_text"],
                "query_type": spec["query_type"],
            }
            sources = build_sources(q_rec_out, tavily_results, next_source_number)
            all_results.extend(sources)
            next_source_number += len(sources)

        if out["oa_sources"]:
            all_results.extend(out["oa_sources"])
        if out["s2_sources"]:
            all_results.extend(out["s2_sources"])

    if not all_results:
        return ResearchPackage([])

    unique_results = deduplicate_sources(all_results)
    source_policy = research_plan.get("source_policy") if isinstance(research_plan, dict) else None
    ranked_sources = rank_sources(
        unique_results,
        research_questions,
        source_policy=source_policy
    )

    selection_bundle = select_sources(
        research_questions,
        ranked_sources
    )
    unique_selected = selection_bundle["unique_selected_sources"]

    retrieval_stats = await asyncio.to_thread(
        retrieve_selected_sources,
        tavily_client,
        unique_selected
    )

    print(
        f"\n[Targeted] Retrieval completed: "
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
        duplicates_merged=len(all_results) - len(unique_results)
    )


def run_targeted_research(
    tavily_client,
    topic,
    targeted_questions,
    gemini_client=None,
    research_plan=None,
    openalex_provider=None,
    semantic_scholar_provider=None
):
    """
    Synchronous entrypoint wrapping the concurrent asyncio.gather() targeted research pass.
    """
    return _run_coro_sync(
        run_targeted_research_async(
            tavily_client=tavily_client,
            topic=topic,
            targeted_questions=targeted_questions,
            gemini_client=gemini_client,
            research_plan=research_plan,
            openalex_provider=openalex_provider,
            semantic_scholar_provider=semantic_scholar_provider,
        )
    )
