import json


# ==========================================================
# ALLOWED VALUES
# ==========================================================

ALLOWED_PRIORITIES = {
    "high",
    "medium",
    "low"
}

ALLOWED_RESEARCH_MODES = {
    "exploratory",
    "comparative",
    "causal",
    "evaluative",
    "descriptive"
}


# ==========================================================
# PLANNER AGENT
# ==========================================================

def create_research_plan(client, model, topic):
    """
    Create a structured research investigation plan.

    The Planner decides:
    - Research domain
    - Research mode
    - Neutral working hypothesis when appropriate
    - Operational/evaluation criteria
    - 3-5 research questions
    - Question type and priority
    - Evidence requirements
    - Preferred source types
    - Quantitative evidence requirements
    - Counter-evidence requirements
    - Boundary conditions
    - Question dependencies

    The Planner DOES NOT:
    - Search the web
    - Select actual sources
    - Rank actual sources
    - Extract facts
    - Verify claims
    - Produce research conclusions
    - Write the final report
    """

    # ======================================================
    # PLANNER PROMPT
    # ======================================================

    prompt = f"""
You are the Planner Agent in an autonomous
research system.

Your responsibility is to DESIGN a rigorous
research investigation.

You must NOT answer the research topic.

You must NOT search for evidence.

You must NOT invent research findings.

You must NOT decide the final conclusion.

RESEARCH TOPIC:

{topic}


==================================================
STEP 1 - IDENTIFY THE RESEARCH DOMAIN
==================================================

Determine the primary research domain.

Examples:

- AI/ML
- Medical
- Economics
- Technology
- Environment
- Science
- Law
- Social Science
- General

Choose the most appropriate domain.


==================================================
STEP 2 - IDENTIFY THE RESEARCH MODE
==================================================

Choose ONE research mode:

- exploratory
- comparative
- causal
- evaluative
- descriptive

Definitions:

exploratory:
Used when the topic mainly seeks broad
understanding.

comparative:
Used when two or more approaches, systems,
methods, groups, or technologies are compared.

causal:
Used when investigating whether one factor
causes or influences another.

evaluative:
Used when assessing effectiveness, quality,
performance, risks, outcomes, or usefulness.

descriptive:
Used when primarily describing the current
state, characteristics, or development of
something.


==================================================
STEP 3 - CREATE A NEUTRAL WORKING HYPOTHESIS
==================================================

If the research mode is:

- comparative
- causal
- evaluative

generate a neutral working hypothesis.

IMPORTANT:

The hypothesis is NOT a conclusion.

It must be:

- neutral
- testable
- falsifiable or qualifiable
- open to supporting evidence
- open to contradictory evidence

It must NOT assume which side is correct.

BAD EXAMPLE:

"Transformer scaling will reach diminishing
returns and alternative architectures will
be necessary."

This assumes the conclusion before research.

BETTER EXAMPLE:

"Continued transformer scaling may produce
further capability gains, but its marginal
technical and economic returns may vary with
compute, data, architecture, task demands,
and deployment constraints."

The evidence collected later must be able to:

- support the hypothesis
- qualify the hypothesis
- contradict the hypothesis
- reject the hypothesis

For exploratory or descriptive research where
a hypothesis would be artificial, return:

null


==================================================
STEP 4 - DEFINE OPERATIONAL / EVALUATION CRITERIA
==================================================

Determine whether the research topic contains
important concepts that are ambiguous,
subjective, broad, or require an operational
definition.

Examples include:

- AGI
- intelligence
- economically useful
- safety
- fairness
- success
- effectiveness
- reliability
- scalability
- sustainability
- high performance

If such concepts exist, define observable or
measurable criteria that later agents can use
to evaluate them.

Example:

For "economically useful AGI", possible
operational criteria could include:

- ability to perform economically valuable tasks
- generalization across multiple task domains
- reliability
- level of human supervision required
- productivity improvement
- deployment cost
- inference cost
- economic value relative to operating cost

IMPORTANT:

These criteria are NOT claimed to be universally
accepted definitions.

They are operational criteria for THIS specific
research investigation.

If no special operational criteria are needed,
return:

[]


==================================================
STEP 5 - CREATE 3 TO 5 RESEARCH QUESTIONS
==================================================

Generate between 3 and 5 focused research
questions.

Each question must investigate a different
important dimension of the original topic.

Possible question types include:

- definition
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

Choose question types appropriate to the
specific research topic.

Do NOT force every topic into the same
question categories.

If STEP 4 identified an important ambiguous
concept, ensure that the investigation
adequately establishes how that concept will
be evaluated.

Questions should form a coherent investigation,
not merely a list of related subtopics.


==================================================
STEP 6 - DEFINE EVIDENCE REQUIREMENTS
==================================================

For EVERY research question, specify what
evidence would actually be needed to answer it.

Examples:

For performance research:

- benchmark results
- accuracy
- latency
- computational cost
- experimental conditions
- baseline comparison

For medical research:

- clinical outcomes
- sample size
- dosage
- adverse effects
- study design
- confidence intervals

For economics:

- economic indicators
- historical data
- household expenditure
- productivity measurements
- employment statistics
- cost data

These are examples only.

Generate evidence requirements appropriate
to the ACTUAL research question.

Do NOT provide research findings here.


==================================================
STEP 7 - DEFINE PREFERRED SOURCE TYPES
==================================================

For EVERY research question, specify which
TYPES of sources should ideally be collected.

Source preferences must depend on the
research domain.

AI/ML examples:

- peer-reviewed research paper
- conference paper
- primary research preprint
- benchmark paper
- official technical report
- official documentation
- systematic survey

Medical examples:

- systematic review
- meta-analysis
- clinical guideline
- peer-reviewed clinical study
- major medical institution
- public health authority

Economics examples:

- government dataset
- academic research paper
- central bank publication
- international organization report
- official statistical agency

Technology examples:

- peer-reviewed research
- official documentation
- technical standard
- security advisory
- technical report

IMPORTANT:

Do NOT provide actual URLs.

Do NOT provide specific paper names.

Do NOT pretend that sources have already been
found.

Actual source discovery belongs to the
Search Agent in Phase 3.


==================================================
STEP 8 - QUANTITATIVE EVIDENCE REQUIREMENT
==================================================

For EVERY question determine whether answering
it properly requires numerical or measurable
evidence.

Return:

true

or

false

Examples requiring quantitative evidence:

- performance comparisons
- cost comparisons
- benchmark comparisons
- medical risk rates
- economic impact
- latency
- accuracy
- prevalence
- energy consumption
- productivity changes


==================================================
STEP 9 - COUNTER-EVIDENCE REQUIREMENT
==================================================

For EVERY question determine whether later
agents should explicitly search for evidence
that challenges, contradicts, or qualifies
the expected interpretation.

Set:

requires_counter_evidence = true

when appropriate for:

- comparisons
- performance claims
- causal claims
- effectiveness claims
- controversial claims
- risks
- strong hypotheses

Counter-evidence means evidence capable of
challenging the current interpretation.

It does NOT mean simply finding another source
that supports the same conclusion.


==================================================
STEP 10 - QUESTION-SPECIFIC BOUNDARY CONDITIONS
==================================================

For EVERY research question identify conditions
under which the ANSWER TO THAT SPECIFIC
QUESTION could change.

Ask:

"Under what conditions might the answer to
this particular research question be
different?"

Boundary conditions must NOT simply be general
problems associated with the overall topic.

Example:

For a model-performance question:

GOOD boundary conditions:

- model size
- benchmark type
- training-data regime
- inference compute budget
- architecture
- evaluation methodology

BAD boundary conditions:

- semiconductor shortages
- electricity prices

unless the specific question is about
deployment economics.

For a medical-effectiveness question:

GOOD:

- patient age
- dosage
- treatment duration
- disease severity
- population characteristics

For an economic-impact question:

GOOD:

- country
- industry
- labor costs
- deployment scale
- time period
- macroeconomic conditions

If no meaningful boundary condition exists,
return:

[]


==================================================
STEP 11 - QUESTION DEPENDENCIES
==================================================

Determine whether answering one research
question logically requires findings from
another question.

Example investigation:

Q1 defines evaluation criteria.

Q2 establishes baseline evidence.

Q3 compares alternatives using that baseline.

Q4 investigates limitations and
counter-evidence.

Q5 evaluates the overall hypothesis.

Possible dependencies:

Q1: []

Q2: ["Q1"]

Q3: ["Q1", "Q2"]

Q4: ["Q2"]

Q5: ["Q1", "Q2", "Q3", "Q4"]

IMPORTANT:

Do NOT create dependencies merely for the
sake of having them.

Use:

[]

when a question can be researched
independently.


==================================================
GENERAL REQUIREMENTS
==================================================

1. Generate only 3 to 5 questions.

2. Every question must be directly relevant
   to the original topic.

3. Avoid duplicate or highly overlapping
   questions.

4. Questions must be specific enough for a
   Search Agent to research.

5. Together the questions should provide
   strong coverage of the topic.

6. Include limitations, boundary conditions,
   risks, or counter-evidence when appropriate.

7. Do NOT answer any research question.

8. Do NOT provide research findings.

9. Do NOT fabricate citations.

10. Do NOT fabricate source names.

11. Priority must be exactly one of:

high
medium
low

12. Return ONLY valid JSON.

13. Do not include Markdown.

14. Do not include ```json.

15. Do not include explanations before or
    after the JSON.


==================================================
REQUIRED JSON STRUCTURE
==================================================

{{
    "topic": "{topic}",

    "domain": "Detected domain",

    "research_mode": "comparative",

    "hypothesis": "Neutral working hypothesis or null",

    "evaluation_criteria": [
        "Criterion 1",
        "Criterion 2"
    ],

    "questions": [
        {{
            "id": "Q1",

            "type": "performance",

            "priority": "high",

            "question": "Research question?",

            "evidence_needed": [
                "Evidence requirement 1",
                "Evidence requirement 2"
            ],

            "preferred_source_types": [
                "Source type 1",
                "Source type 2"
            ],

            "requires_quantitative_evidence": true,

            "requires_counter_evidence": true,

            "boundary_conditions": [
                "Question-specific condition 1",
                "Question-specific condition 2"
            ],

            "depends_on": []
        }},

        {{
            "id": "Q2",

            "type": "limitations",

            "priority": "medium",

            "question": "Research question?",

            "evidence_needed": [
                "Evidence requirement"
            ],

            "preferred_source_types": [
                "Source type"
            ],

            "requires_quantitative_evidence": false,

            "requires_counter_evidence": true,

            "boundary_conditions": [],

            "depends_on": ["Q1"]
        }}
    ]
}}
"""

    # ======================================================
    # CALL GEMINI
    # ======================================================

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    if not response.text:
        raise RuntimeError(
            "Planner Agent returned an empty response."
        )

    raw_output = response.text.strip()


    # ======================================================
    # PARSE JSON
    # ======================================================

    try:
        plan = json.loads(
            raw_output
        )

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Planner Agent did not return valid JSON.\n"
            f"Received:\n{raw_output}"
        ) from error


    # ======================================================
    # VALIDATE TOP-LEVEL STRUCTURE
    # ======================================================

    if not isinstance(
        plan,
        dict
    ):
        raise RuntimeError(
            "Planner output must be a JSON object."
        )

    required_top_fields = {
        "topic",
        "domain",
        "research_mode",
        "hypothesis",
        "evaluation_criteria",
        "questions"
    }

    missing_top_fields = (
        required_top_fields
        - plan.keys()
    )

    if missing_top_fields:
        raise RuntimeError(
            "Planner output is missing: "
            + ", ".join(
                sorted(missing_top_fields)
            )
        )


    # ======================================================
    # VALIDATE DOMAIN
    # ======================================================

    domain = plan[
        "domain"
    ]

    if not isinstance(
        domain,
        str
    ):
        raise RuntimeError(
            "Planner domain must be text."
        )

    domain = domain.strip()

    if not domain:
        raise RuntimeError(
            "Planner returned an empty domain."
        )


    # ======================================================
    # VALIDATE RESEARCH MODE
    # ======================================================

    research_mode = str(
        plan[
            "research_mode"
        ]
    ).strip().lower()

    if (
        research_mode
        not in ALLOWED_RESEARCH_MODES
    ):
        raise RuntimeError(
            "Invalid research mode: "
            f"{research_mode}"
        )


    # ======================================================
    # VALIDATE HYPOTHESIS
    # ======================================================

    hypothesis = plan[
        "hypothesis"
    ]

    if hypothesis is not None:

        if not isinstance(
            hypothesis,
            str
        ):
            raise RuntimeError(
                "Hypothesis must be text or null."
            )

        hypothesis = (
            hypothesis.strip()
        )

        if not hypothesis:
            hypothesis = None


    # ======================================================
    # VALIDATE EVALUATION CRITERIA
    # ======================================================

    evaluation_criteria = plan[
        "evaluation_criteria"
    ]

    if not isinstance(
        evaluation_criteria,
        list
    ):
        raise RuntimeError(
            "'evaluation_criteria' must be a list."
        )

    evaluation_criteria = [
        str(value).strip()
        for value
        in evaluation_criteria
        if str(value).strip()
    ]


    # ======================================================
    # VALIDATE QUESTIONS
    # ======================================================

    questions = plan[
        "questions"
    ]

    if not isinstance(
        questions,
        list
    ):
        raise RuntimeError(
            "'questions' must be a list."
        )

    if not 3 <= len(questions) <= 5:
        raise RuntimeError(
            "Planner must generate between "
            "3 and 5 questions."
        )


    # ======================================================
    # REQUIRED QUESTION FIELDS
    # ======================================================

    required_question_fields = {
        "id",
        "type",
        "priority",
        "question",
        "evidence_needed",
        "preferred_source_types",
        "requires_quantitative_evidence",
        "requires_counter_evidence",
        "boundary_conditions",
        "depends_on"
    }


    # ======================================================
    # CLEAN QUESTIONS
    # ======================================================

    cleaned_questions = []

    seen_ids = set()


    for index, item in enumerate(
        questions,
        start=1
    ):

        # --------------------------------------------------
        # Validate question object
        # --------------------------------------------------

        if not isinstance(
            item,
            dict
        ):
            raise RuntimeError(
                f"Question {index} must be "
                "a JSON object."
            )


        # --------------------------------------------------
        # Check required fields
        # --------------------------------------------------

        missing_fields = (
            required_question_fields
            - item.keys()
        )

        if missing_fields:
            raise RuntimeError(
                f"Question {index} is missing: "
                + ", ".join(
                    sorted(missing_fields)
                )
            )


        # --------------------------------------------------
        # Validate ID
        # --------------------------------------------------

        question_id = str(
            item[
                "id"
            ]
        ).strip().upper()

        if not question_id:
            raise RuntimeError(
                f"Question {index} has no ID."
            )

        if question_id in seen_ids:
            raise RuntimeError(
                "Duplicate question ID: "
                f"{question_id}"
            )

        seen_ids.add(
            question_id
        )


        # --------------------------------------------------
        # Validate question type
        # --------------------------------------------------

        question_type = str(
            item[
                "type"
            ]
        ).strip().lower()

        if not question_type:
            raise RuntimeError(
                f"Question {index} has no type."
            )


        # --------------------------------------------------
        # Validate priority
        # --------------------------------------------------

        priority = str(
            item[
                "priority"
            ]
        ).strip().lower()

        if (
            priority
            not in ALLOWED_PRIORITIES
        ):
            raise RuntimeError(
                f"Question {index} has invalid "
                f"priority: {priority}"
            )


        # --------------------------------------------------
        # Validate question text
        # --------------------------------------------------

        question_text = str(
            item[
                "question"
            ]
        ).strip()

        if not question_text:
            raise RuntimeError(
                f"Question {index} is empty."
            )


        # --------------------------------------------------
        # Validate evidence requirements
        # --------------------------------------------------

        evidence_needed = item[
            "evidence_needed"
        ]

        if not isinstance(
            evidence_needed,
            list
        ):
            raise RuntimeError(
                f"Question {index}: "
                "'evidence_needed' must be a list."
            )

        evidence_needed = [
            str(value).strip()
            for value
            in evidence_needed
            if str(value).strip()
        ]

        if not evidence_needed:
            raise RuntimeError(
                f"Question {index} must specify "
                "at least one evidence requirement."
            )


        # --------------------------------------------------
        # Validate preferred source types
        # --------------------------------------------------

        preferred_source_types = item[
            "preferred_source_types"
        ]

        if not isinstance(
            preferred_source_types,
            list
        ):
            raise RuntimeError(
                f"Question {index}: "
                "'preferred_source_types' "
                "must be a list."
            )

        preferred_source_types = [
            str(value).strip()
            for value
            in preferred_source_types
            if str(value).strip()
        ]

        if not preferred_source_types:
            raise RuntimeError(
                f"Question {index} must specify "
                "preferred source types."
            )


        # --------------------------------------------------
        # Validate quantitative evidence flag
        # --------------------------------------------------

        quantitative = item[
            "requires_quantitative_evidence"
        ]

        if not isinstance(
            quantitative,
            bool
        ):
            raise RuntimeError(
                f"Question {index}: "
                "'requires_quantitative_evidence' "
                "must be true or false."
            )


        # --------------------------------------------------
        # Validate counter-evidence flag
        # --------------------------------------------------

        counter_evidence = item[
            "requires_counter_evidence"
        ]

        if not isinstance(
            counter_evidence,
            bool
        ):
            raise RuntimeError(
                f"Question {index}: "
                "'requires_counter_evidence' "
                "must be true or false."
            )


        # --------------------------------------------------
        # Validate boundary conditions
        # --------------------------------------------------

        boundary_conditions = item[
            "boundary_conditions"
        ]

        if not isinstance(
            boundary_conditions,
            list
        ):
            raise RuntimeError(
                f"Question {index}: "
                "'boundary_conditions' "
                "must be a list."
            )

        boundary_conditions = [
            str(value).strip()
            for value
            in boundary_conditions
            if str(value).strip()
        ]


        # --------------------------------------------------
        # Validate dependencies
        # --------------------------------------------------

        depends_on = item[
            "depends_on"
        ]

        if not isinstance(
            depends_on,
            list
        ):
            raise RuntimeError(
                f"Question {index}: "
                "'depends_on' must be a list."
            )

        depends_on = [
            str(value).strip().upper()
            for value
            in depends_on
            if str(value).strip()
        ]


        # --------------------------------------------------
        # Store cleaned question
        # --------------------------------------------------

        cleaned_questions.append(
            {
                "id":
                    question_id,

                "type":
                    question_type,

                "priority":
                    priority,

                "question":
                    question_text,

                "evidence_needed":
                    evidence_needed,

                "preferred_source_types":
                    preferred_source_types,

                "requires_quantitative_evidence":
                    quantitative,

                "requires_counter_evidence":
                    counter_evidence,

                "boundary_conditions":
                    boundary_conditions,

                "depends_on":
                    depends_on
            }
        )


    # ======================================================
    # VALIDATE DEPENDENCIES
    # ======================================================

    valid_ids = {
        question["id"]
        for question
        in cleaned_questions
    }

    for question in cleaned_questions:

        for dependency in question[
            "depends_on"
        ]:

            # ----------------------------------------------
            # Dependency must exist
            # ----------------------------------------------

            if dependency not in valid_ids:
                raise RuntimeError(
                    f"{question['id']} depends on "
                    f"unknown question {dependency}."
                )

            # ----------------------------------------------
            # Question cannot depend on itself
            # ----------------------------------------------

            if (
                dependency
                == question["id"]
            ):
                raise RuntimeError(
                    f"{question['id']} cannot "
                    "depend on itself."
                )


    # ======================================================
    # RETURN FINAL STRUCTURED PLAN
    # ======================================================

    return {
        "topic":
            topic,

        "domain":
            domain,

        "research_mode":
            research_mode,

        "hypothesis":
            hypothesis,

        "evaluation_criteria":
            evaluation_criteria,

        "questions":
            cleaned_questions
    }