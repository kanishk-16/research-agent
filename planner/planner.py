import json

from google.genai import types

from .prompt import build_planner_prompt
from .schema import build_planner_schema

from .validation import validate_plan
from .validation.helpers import clean_string


def create_research_plan(
    client,
    model,
    topic,
):
    """
    Create a machine-executable research plan.

    Pipeline:

    Topic
      -> Planner Prompt
      -> Gemini Structured Output
      -> JSON Parsing
      -> Logical Validation
      -> Normalized Research Plan

    The Planner does NOT:
    - search the web
    - find actual sources
    - provide actual URLs
    - extract evidence
    - verify citations
    - execute research questions
    - answer the research topic
    - produce the final conclusion
    """

    # ======================================================
    # 1. VALIDATE INPUT
    # ======================================================

    topic = clean_string(topic)

    if not topic:
        raise ValueError(
            "Research topic cannot be empty."
        )

    # ======================================================
    # 2. BUILD PROMPT
    # ======================================================

    prompt = build_planner_prompt(
        topic
    )

    # ======================================================
    # 3. BUILD OUTPUT SCHEMA
    # ======================================================

    response_schema = build_planner_schema()

    # ======================================================
    # 4. CALL GEMINI
    # ======================================================

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,

            config=types.GenerateContentConfig(
                temperature=0.1,

                response_mime_type=(
                    "application/json"
                ),

                response_schema=response_schema,
            ),
        )

    except Exception as exc:
        raise RuntimeError(
            "Planner Agent model call failed.\n"
            f"Reason: {exc}"
        ) from exc

    # ======================================================
    # 5. CHECK RESPONSE
    # ======================================================

    if not response.text:
        raise RuntimeError(
            "Planner Agent returned an empty response."
        )

    raw_output = response.text.strip()

    # ======================================================
    # 6. PARSE JSON
    # ======================================================

    try:
        plan = json.loads(
            raw_output
        )

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Planner Agent did not return "
            "valid JSON.\n\n"
            "Raw response:\n"
            f"{raw_output}"
        ) from exc

    # ======================================================
    # 7. VALIDATE + NORMALIZE
    # ======================================================

    try:
        validated_plan = validate_plan(
            plan=plan,
            topic=topic,
        )

    except Exception as exc:
        formatted_plan = json.dumps(
            plan,
            indent=2,
            ensure_ascii=False,
        )

        raise RuntimeError(
            "Planner Agent returned structured JSON, "
            "but the research plan failed validation."
            "\n\n"
            "Validation error:\n"
            f"{exc}"
            "\n\n"
            "JSON returned by Planner:\n"
            f"{formatted_plan}"
        ) from exc

    # ======================================================
    # 8. RETURN PLAN
    # ======================================================

    return validated_plan