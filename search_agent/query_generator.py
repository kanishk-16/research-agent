import json


MODEL = "gemini-3.5-flash-lite"

NORMAL_QUERIES_PER_QUESTION = 2
COUNTER_QUERIES_PER_QUESTION = 1


def generate_search_queries(
    gemini_client,
    research_question
):
    """
    Generate bounded normal and counter-evidence
    search queries for one Planner question.
    """

    question_id = str(
        research_question.get(
            "id",
            "Q1"
        )
    ).strip().upper()

    question_text = str(
        research_question.get(
            "question",
            ""
        )
    ).strip()

    requires_counter_evidence = (
        research_question.get(
            "requires_counter_evidence",
            False
        )
    )

    prompt = f"""
You generate focused web search queries for one research question.

Research question ID: {question_id}
Research question: {question_text}

Evidence requirements:
{research_question.get("evidence_needed", [])}

Preferred source types:
{research_question.get("preferred_source_types", [])}

Boundary conditions:
{research_question.get("boundary_conditions", [])}

Counter-evidence required:
{requires_counter_evidence}

Return ONLY valid JSON with this exact structure:
{{
  "normal_queries": ["...", "..."],
  "counter_queries": ["..."]
}}

Generate exactly {NORMAL_QUERIES_PER_QUESTION} concise,
distinct, evidence-oriented normal queries.
Generate exactly {COUNTER_QUERIES_PER_QUESTION} concise counter-evidence
query when counter-evidence is required; otherwise return an empty list.
Do not include URLs, source names, citations, explanations, or Markdown.
"""

    response = gemini_client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    if not response.text:
        raise RuntimeError(
            f"Query generation returned an empty response for {question_id}."
        )

    try:
        generated = json.loads(
            response.text.strip()
        )

    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Query generation returned invalid JSON for {question_id}."
        ) from error

    if not isinstance(generated, dict):
        raise RuntimeError(
            f"Query generation must return an object for {question_id}."
        )

    normal_queries = generated.get(
        "normal_queries"
    )

    counter_queries = generated.get(
        "counter_queries"
    )

    if not isinstance(normal_queries, list):
        raise RuntimeError(
            f"Normal queries must be a list for {question_id}."
        )

    if not isinstance(counter_queries, list):
        raise RuntimeError(
            f"Counter queries must be a list for {question_id}."
        )

    def clean_queries(
        values,
        query_type,
        maximum
    ):
        cleaned = []
        seen = set()

        for value in values:

            if not isinstance(value, str):
                raise RuntimeError(
                    f"{query_type} query must be text for {question_id}."
                )

            query_text = value.strip()
            query_key = query_text.casefold()

            if not query_text or query_key in seen:
                continue

            seen.add(query_key)
            cleaned.append(query_text)

        if len(cleaned) > maximum:
            raise RuntimeError(
                f"Too many {query_type} queries for {question_id}."
            )

        return cleaned

    normal_queries = clean_queries(
        normal_queries,
        "normal",
        NORMAL_QUERIES_PER_QUESTION
    )

    counter_queries = clean_queries(
        counter_queries,
        "counter",
        COUNTER_QUERIES_PER_QUESTION
    )

    if len(normal_queries) != NORMAL_QUERIES_PER_QUESTION:
        raise RuntimeError(
            f"Expected {NORMAL_QUERIES_PER_QUESTION} normal queries "
            f"for {question_id}."
        )

    if requires_counter_evidence and (
        len(counter_queries)
        != COUNTER_QUERIES_PER_QUESTION
    ):
        raise RuntimeError(
            f"Expected {COUNTER_QUERIES_PER_QUESTION} counter query "
            f"for {question_id}."
        )

    if not requires_counter_evidence:
        counter_queries = []

    return [
        {
            "question_id": question_id,
            "query_text": query_text,
            "query_type": "normal"
        }
        for query_text in normal_queries
    ] + [
        {
            "question_id": question_id,
            "query_text": query_text,
            "query_type": "counter"
        }
        for query_text in counter_queries
    ]


def fallback_search_queries(
    research_question
):
    """
    Create bounded queries when Gemini query generation fails.
    """

    question_id = str(
        research_question.get(
            "id",
            "Q1"
        )
    ).strip().upper()

    question_text = str(
        research_question.get(
            "question",
            ""
        )
    ).strip()

    queries = [
        {
            "question_id": question_id,
            "query_text": question_text,
            "query_type": "normal"
        }
    ]

    if research_question.get(
        "requires_counter_evidence",
        False
    ):
        queries.append(
            {
                "question_id": question_id,
                "query_text": (
                    f"{question_text} limitations failures "
                    "contradictory evidence"
                ),
                "query_type": "counter"
            }
        )

    return queries
