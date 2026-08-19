import json


ALLOWED_PRIORITIES = {
    "high",
    "medium",
    "low"
}


def create_research_plan(client, model, topic):
    """
    Planner Agent

    Creates a structured investigation plan
    for the given research topic.

    Returns:
        {
            "topic": "...",
            "domain": "...",
            "questions": [...]
        }
    """

    prompt = f"""
You are the Planner Agent in an autonomous
research system.

Your job is NOT to answer the topic.

Your job is to design a strong investigation
strategy for researching the topic.

RESEARCH TOPIC:
{topic}

First, identify the main research domain.

Examples of domains:
- AI/ML
- Medical
- Economics
- Technology
- Environment
- Science
- Law
- Social Science
- General

Then generate between 3 and 5 focused
research questions.

Each research question must have:

1. id
2. type
3. question
4. priority

Possible question types include:

- background
- current_state
- performance
- comparison
- quantitative
- causes
- effects
- risks
- limitations
- boundary_conditions
- counter_evidence
- economics
- social_impact
- production
- future

Choose question types that make sense for
the specific research topic.

Requirements:

1. Every question must be directly relevant
   to the original topic.

2. Each question should investigate a
   different aspect of the topic.

3. Avoid duplicate or highly overlapping
   questions.

4. Questions must be specific enough to
   search on the web.

5. Together, the questions should provide
   strong coverage of the topic.

6. Include a limitations, boundary_conditions,
   risks, or counter_evidence question when
   appropriate.

7. Do NOT answer any research question.

8. Generate only 3 to 5 questions.

9. Priority must be one of:
   high, medium, low

10. Return ONLY valid JSON.

Required JSON structure:

{{
    "topic": "{topic}",
    "domain": "Detected domain",
    "questions": [
        {{
            "id": "Q1",
            "type": "question_type",
            "question": "Research question?",
            "priority": "high"
        }},
        {{
            "id": "Q2",
            "type": "question_type",
            "question": "Research question?",
            "priority": "medium"
        }}
    ]
}}

Do not include markdown.
Do not include ```json.
Do not include explanations before or after
the JSON.
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    if not response.text:
        raise RuntimeError(
            "Planner Agent returned an empty response."
        )

    raw_output = response.text.strip()

    # ------------------------------------------
    # Convert JSON text into Python dictionary
    # ------------------------------------------

    try:
        plan = json.loads(raw_output)

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Planner Agent did not return valid JSON.\n"
            f"Received:\n{raw_output}"
        ) from error

    # ------------------------------------------
    # Validate top-level structure
    # ------------------------------------------

    if not isinstance(plan, dict):
        raise RuntimeError(
            "Planner output must be a JSON object."
        )

    if "topic" not in plan:
        raise RuntimeError(
            "Planner output is missing 'topic'."
        )

    if "domain" not in plan:
        raise RuntimeError(
            "Planner output is missing 'domain'."
        )

    if "questions" not in plan:
        raise RuntimeError(
            "Planner output is missing 'questions'."
        )

    # ------------------------------------------
    # Validate domain
    # ------------------------------------------

    domain = plan["domain"]

    if not isinstance(domain, str):
        raise RuntimeError(
            "Planner domain must be text."
        )

    domain = domain.strip()

    if not domain:
        raise RuntimeError(
            "Planner returned an empty domain."
        )

    # ------------------------------------------
    # Validate questions
    # ------------------------------------------

    questions = plan["questions"]

    if not isinstance(questions, list):
        raise RuntimeError(
            "'questions' must be a list."
        )

    if not 3 <= len(questions) <= 5:
        raise RuntimeError(
            "Planner must generate between "
            "3 and 5 questions."
        )

    cleaned_questions = []

    for index, item in enumerate(
        questions,
        start=1
    ):

        if not isinstance(item, dict):
            raise RuntimeError(
                f"Question {index} must be "
                "a JSON object."
            )

        required_fields = {
            "id",
            "type",
            "question",
            "priority"
        }

        missing_fields = (
            required_fields - item.keys()
        )

        if missing_fields:
            raise RuntimeError(
                f"Question {index} is missing: "
                f"{', '.join(missing_fields)}"
            )

        question_id = str(
            item["id"]
        ).strip()

        question_type = str(
            item["type"]
        ).strip()

        question_text = str(
            item["question"]
        ).strip()

        priority = str(
            item["priority"]
        ).strip().lower()

        if not question_text:
            raise RuntimeError(
                f"Question {index} is empty."
            )

        if not question_type:
            raise RuntimeError(
                f"Question {index} has no type."
            )

        if priority not in ALLOWED_PRIORITIES:
            raise RuntimeError(
                f"Question {index} has invalid "
                f"priority: {priority}"
            )

        cleaned_questions.append(
            {
                "id": question_id,
                "type": question_type,
                "question": question_text,
                "priority": priority
            }
        )

    # ------------------------------------------
    # Return clean structured plan
    # ------------------------------------------

    return {
        "topic": topic,
        "domain": domain,
        "questions": cleaned_questions
    }