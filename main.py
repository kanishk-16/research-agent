import json
import os
import sys
import warnings

# Suppress AFC user warnings from google-genai SDK
warnings.filterwarnings("ignore", category=UserWarning, module="google.genai")

from dotenv import load_dotenv
from google import genai
from tavily import TavilyClient

from planner.planner import create_research_plan
from search_agent.researcher import run_research, run_targeted_research, ResearchPackage
from search_agent.source_ranker import rank_sources, print_ranking_diagnostics
from search_agent.source_selector import select_sources
from search_agent.content_retriever import retrieve_selected_sources
from search_agent.sources import deduplicate_sources
from search_agent.evidence_extractor import extract_research_evidence
from search_agent.evidence_output import build_research_evidence_output, save_research_evidence
from search_agent.summary_generator import (
    generate_research_summary,
    detect_empirical_contradictions,
)
from search_agent.source_providers import (
    OpenAlexProvider,
    SemanticScholarProvider,
)
from search_agent.evidence_sufficiency import (
    check_all_sufficiency,
    generate_targeted_queries,
)

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

MODEL = "gemini-3.8-flash"

SEARCH_DEPTH = "advanced"

MAX_RESULTS_PER_QUERY = 5

PLAN_OUTPUT_FILE = "data/research_plan.json"
EVIDENCE_OUTPUT_FILE = "data/research_evidence.json"

SEPARATOR = "=" * 70
SUB_SEPARATOR = "-" * 70


# ==========================================================
# CLIENT SETUP & CIRCUIT-BREAKER ROUTING LAYER
# ==========================================================

from dataclasses import dataclass, field
import inspect
import random
import time

FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

_ALLOWED_FREE_TIER_MODELS = set(FALLBACK_MODELS)

CHAINS = {
    "planner": [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ],
    "extractor": [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
    ],
    "synth": [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ],
}


@dataclass
class Breaker:
    fail_threshold: int = 2
    cooldown: float = 60.0
    fails: dict = field(default_factory=dict)
    open_until: dict = field(default_factory=dict)

    def available(self, m: str) -> bool:
        return time.time() >= self.open_until.get(m, 0.0)

    def record_fail(self, m: str, permanent: bool = False):
        if permanent:
            self.open_until[m] = float("inf")
            self.fails[m] = 0
            return
        self.fails[m] = self.fails.get(m, 0) + 1
        if self.fails[m] >= self.fail_threshold:
            self.open_until[m] = time.time() + self.cooldown
            self.fails[m] = 0

    def record_ok(self, m: str):
        self.fails[m] = 0
        self.open_until.pop(m, None)


# Module-level circuit breaker shared by planner, extractor, critic, verifier, and synthesizer
breaker = Breaker()


def classify(e: Exception) -> str:
    """
    Classify model exceptions into:
    - 'skip': model retired (404), permanent zero-quota (limit: 0), or unsupported feature
    - 'retry': transient server overload (503, 500, 504) or rate limit (429 with retry delay)
    - 'fatal': bad request or auth/config errors (400, 401, 403)
    """
    s = str(e)
    if any(c in s for c in ("404", "NOT_FOUND")) or "limit: 0" in s:
        return "skip"
    if any(
        c in s
        for c in (
            "503",
            "UNAVAILABLE",
            "overloaded",
            "500",
            "INTERNAL",
            "504",
            "DEADLINE_EXCEEDED",
            "429",
            "RESOURCE_EXHAUSTED",
        )
    ):
        return "retry"
    return "fatal"


def route(models, call, retries: int = 1, deadline: float = 60.0):
    """
    Route an LLM call across a role-specific model chain with circuit breaker,
    error classification, brief per-model retry + jitter, and a per-round deadline.
    """
    start, last = time.time(), None
    for m in models:
        if not breaker.available(m):
            continue
        for attempt in range(retries + 1):
            if time.time() - start > deadline:
                raise TimeoutError(f"Deadline ({deadline}s) hit. Last error: {last}")
            try:
                out = call(m)
                breaker.record_ok(m)
                return out
            except Exception as e:
                last, kind = e, classify(e)
                if kind == "fatal":
                    raise
                if kind == "skip":
                    breaker.record_fail(m, permanent=True)
                    break
                breaker.record_fail(m)
                if attempt < retries:
                    time.sleep(min(2 ** attempt + random.random(), 8.0))
    raise RuntimeError(
        f"Provider overloaded across all fallback models ({', '.join(models)}). "
        f"Please try again in a few minutes. Last error: {last}"
    ) from last


def route_patient(
    models,
    call,
    retries: int = 1,
    round_deadline: float = 60.0,
    total_deadline: float = 180.0,
    wait: float = 20.0,
):
    """
    Interactive patient router: if an entire model family experiences a simultaneous
    503 spike, pause for `wait` seconds (growing 1.5x up to 45s), clear temporary
    cooldowns, and retry the chain up to `total_deadline` seconds total.
    """
    start = time.time()
    current_wait = float(wait)
    while True:
        # Ensure temporary cooldowns don't block the entire chain at the start of a patient round
        if not any(breaker.available(m) for m in models):
            for m in models:
                if breaker.open_until.get(m, 0.0) != float("inf"):
                    breaker.open_until.pop(m, None)
        try:
            return route(models, call, retries=retries, deadline=round_deadline)
        except (RuntimeError, TimeoutError) as e:
            if time.time() - start + current_wait > total_deadline:
                raise RuntimeError(
                    "Provider overloaded (503 high demand across all Gemini Flash models). "
                    "Please try again in a few minutes or add a secondary provider."
                ) from e
            print(f"All models busy, retrying in {int(current_wait)}s...")
            time.sleep(current_wait)
            current_wait = min(current_wait * 1.5, 45.0)


def _infer_caller_role(explicit_role=None) -> str:
    if explicit_role in CHAINS:
        return explicit_role
    try:
        for frame_info in inspect.stack()[1:7]:
            fname = (frame_info.filename or "").replace("\\", "/").lower()
            if any(k in fname for k in ("evidence_extractor", "query_generator", "evidence_sufficiency")):
                return "extractor"
            if any(k in fname for k in ("calibrated_synthesizer", "summary_generator", "claim_verifier")):
                return "synth"
            if any(k in fname for k in ("planner", "plan_critic")):
                return "planner"
    except Exception:
        pass
    return "planner"


class _DualModeModels:
    def __init__(self, std_client, vertex_client):
        self._std = std_client
        self._vertex = vertex_client
        self._active = std_client

    def _call_single(self, *args, **kwargs):
        try:
            return self._active.models.generate_content(*args, **kwargs)
        except Exception as err1:
            err1_str = str(err1)
            if any(k in err1_str for k in ("400", "401", "403", "API_KEY_INVALID", "PERMISSION_DENIED", "INVALID_ARGUMENT")):
                alt = self._vertex if self._active is self._std else self._std
                try:
                    res = alt.models.generate_content(*args, **kwargs)
                    self._active = alt
                    return res
                except Exception:
                    raise err1
            raise

    def generate_content(self, *args, **kwargs):
        call_kwargs = dict(kwargs)
        role = _infer_caller_role(call_kwargs.pop("role", None))
        retries = int(call_kwargs.pop("retries", 1))
        round_deadline = float(call_kwargs.pop("deadline", 60.0))
        total_deadline = float(call_kwargs.pop("total_deadline", 180.0))
        wait = float(call_kwargs.pop("wait", 20.0))

        chain = list(CHAINS.get(role, FALLBACK_MODELS))
        req_model = call_kwargs.get("model")
        if req_model and req_model in _ALLOWED_FREE_TIER_MODELS and role != "extractor":
            chain = [req_model] + [m for m in chain if m != req_model]

        def _do_call(target_model: str):
            kw = dict(call_kwargs)
            kw["model"] = target_model
            return self._call_single(*args, **kw)

        return route_patient(
            chain,
            _do_call,
            retries=retries,
            round_deadline=round_deadline,
            total_deadline=total_deadline,
            wait=wait,
        )

    def embed_content(self, *args, **kwargs):
        try:
            return self._active.models.embed_content(*args, **kwargs)
        except Exception as err1:
            err1_str = str(err1)
            if any(k in err1_str for k in ("400", "401", "403", "API_KEY_INVALID", "PERMISSION_DENIED", "INVALID_ARGUMENT")):
                alt = self._vertex if self._active is self._std else self._std
                try:
                    res = alt.models.embed_content(*args, **kwargs)
                    self._active = alt
                    return res
                except Exception:
                    raise err1
            raise


class _DualModeGeminiClient:
    def __init__(self, api_key: str):
        fast_http = {"retry_options": {"attempts": 1}}
        std_client = genai.Client(api_key=api_key, http_options=fast_http)
        vertex_client = genai.Client(vertexai=True, api_key=api_key, http_options=fast_http)
        self.models = _DualModeModels(std_client, vertex_client)


def load_clients():
    

    load_dotenv()

    gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip().strip('"').strip("'")

    tavily_key = (os.getenv("TAVILY_API_KEY") or "").strip().strip('"').strip("'")

    if not gemini_key:
        raise ValueError(
            "GEMINI_API_KEY is missing from .env"
        )

    if not tavily_key:
        raise ValueError(
            "TAVILY_API_KEY is missing from .env"
        )

    gemini_client = _DualModeGeminiClient(gemini_key)

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
# PHASE 2 - RESEARCHER
# Search + source ranking pipeline
# ==========================================================


# ----------------------------------------------------------
# PHASE 2 - STEP 1
# Search the web (compatibility wrapper)
# ----------------------------------------------------------

def search_web(
    tavily_client,
    topic,
    research_questions=None,
    gemini_client=None,
    research_plan=None,
    openalex_provider=None,
    semantic_scholar_provider=None
):
    """Compatibility wrapper for the Phase 2 researcher."""

    return run_research(
        tavily_client,
        topic,
        research_questions,
        gemini_client,
        research_plan=research_plan,
        openalex_provider=openalex_provider,
        semantic_scholar_provider=semantic_scholar_provider
    )


# ----------------------------------------------------------
# PHASE 2 - STEP 2
# Build research context
# ----------------------------------------------------------

def build_research_context(
    search_results
):
    """
    Convert ranked source objects into
    context that can be given to Gemini.

    This represents the AUGMENTATION step
    in the RAG-style pipeline.
    """

    context = ""

    for index, result in enumerate(
        search_results,
        start=1
    ):

        source_id = result.get(
            "source_id",
            f"S{index}"
        )

        context += f"""
SOURCE {source_id}

Title:
{result.get("title", "Unknown")}

URL:
{result.get("url", "")}

Content:
{result.get("content", "")}

"""

    return context


# ----------------------------------------------------------
# PHASE 2 - STEP 3
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

5. Cite information using the source IDs:
    [Source S1], [Source S2], [Source S3], etc.

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
# PHASE 2 - STEP 4
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

    for result in search_results:

        title = result.get(
            "title",
            "Unknown"
        )

        url = result.get(
            "url",
            ""
        )

        print(
            f"\n[{result.get('source_id', 'Source')}] {title}"
        )

        meta_parts = []
        authors = result.get("authors") or []
        if authors:
            if len(authors) > 3:
                meta_parts.append(", ".join(authors[:3]) + " et al.")
            else:
                meta_parts.append(", ".join(authors))
        if result.get("publication_year"):
            meta_parts.append(str(result["publication_year"]))
        if result.get("venue"):
            meta_parts.append(result["venue"])
        if result.get("citation_count") is not None:
            meta_parts.append(f"Citations: {result['citation_count']}")
        if result.get("source_type"):
            meta_parts.append(f"Type: {result['source_type']}")
        if meta_parts:
            print(f"  {' | '.join(meta_parts)}")

        if result.get("doi"):
            print(f"  DOI: {result['doi']}")

        print(f"  {url}")


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

    dirname = os.path.dirname(PLAN_OUTPUT_FILE)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

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

    ui = None
    try:
        # Check demo and verbose mode
        is_demo = "--demo" in sys.argv
        is_verbose = ("--verbose" in sys.argv or "-v" in sys.argv)
        cli_args = [arg for arg in sys.argv[1:] if arg not in ("--verbose", "-v", "--demo")]

        if is_demo:
            from demo import run_demo
            run_demo(verbose=is_verbose)
            return

        from cli.console import ARAConsole
        from search_agent.source_providers import set_provider_event_hook

        ui = ARAConsole(verbose=is_verbose)
        set_provider_event_hook(ui.show_provider_warning)

        # ==================================================
        # INITIALIZE CLIENTS
        # ==================================================

        (
            gemini_client,
            tavily_client
        ) = load_clients()

        openalex_provider = None
        try:
            mailto = os.getenv("OPENALEX_MAILTO")
            openalex_provider = OpenAlexProvider(mailto=mailto, max_pages=1)
        except Exception as exc:
            print(f"[Warning] Could not initialize OpenAlex provider: {exc}")

        semantic_scholar_provider = None
        try:
            semantic_scholar_provider = SemanticScholarProvider(tavily_client=tavily_client)
            mode_label = "Authenticated API" if semantic_scholar_provider.api_key else "Autonomous Web Discovery"
            print(f"  [Academic Providers] OpenAlex: Active | Semantic Scholar: Active ({mode_label})")
        except Exception as exc:
            print(f"[Warning] Could not initialize Semantic Scholar provider: {exc}")

        # ==================================================
        # STARTUP & TOPIC INPUT
        # ==================================================

        ui.show_banner()
        ui.show_startup(
            openalex_active=bool(openalex_provider),
            semantic_scholar_active=bool(semantic_scholar_provider),
            tavily_active=bool(tavily_client)
        )

        if cli_args:
            raw_topic = " ".join(cli_args).strip()
            topic = ui.prompt_question(default_value=raw_topic)
        elif os.getenv("RESEARCH_TOPIC"):
            raw_topic = os.getenv("RESEARCH_TOPIC").strip()
            topic = ui.prompt_question(default_value=raw_topic)
        else:
            topic = ui.prompt_question()

        if not topic:
            ui.console.print("[dim]Please enter a research topic to begin.[/]")
            return

        from planner.validation.ambiguity_detector import detect_topic_ambiguity
        is_ambiguous, reason, clarifications = detect_topic_ambiguity(topic)
        if is_ambiguous:
            clarified = ui.prompt_clarification(reason, clarifications)
            if not clarified:
                ui.console.print("\n[dim]Research session cancelled due to topic ambiguity.[/]")
                return
            topic = clarified

        # ==================================================
        # PHASE 1
        # PLANNER
        # ==================================================

        ui.spinner("Planning research...")
        research_plan = run_planner(
            gemini_client,
            topic
        )

        # Save Plan & log details
        display_research_plan(
            research_plan
        )
        save_research_plan(
            research_plan
        )
        ui.show_planning_complete(research_plan)

        # ==================================================
        # PHASE 2 & PHASE 3 - UNIFIED RESEARCH ORCHESTRATOR
        # ==================================================

        print_header("PHASE 2 - RESEARCH PIPELINE")
        from search_agent.orchestrator import ResearchOrchestrator

        def _handle_cli_event(event_name: str, payload: dict):
            if event_name == "discovery_start":
                ui.spinner("Searching academic & web providers...")
            elif event_name == "discovery_complete":
                pkg = payload["research_pkg"]
                print_sources(pkg)
                raw_candidates = len(pkg) + getattr(pkg, "duplicates_merged", 0)
                providers = ["Tavily"]
                if openalex_provider:
                    providers.append("OpenAlex")
                if semantic_scholar_provider:
                    providers.append("Semantic Scholar")
                ui.show_search_complete(raw_candidates, len(pkg), providers=providers)
                sel_bundle = getattr(pkg, "selection_bundle", {})
                sel_count = len(sel_bundle.get("unique_selected_sources", []))
                ui.show_ranking_complete(sel_count)
                ret_stats = getattr(pkg, "retrieval_stats", {})
                ui.show_retrieval_complete(ret_stats.get("successes", sel_count), sel_count)
            elif event_name == "extraction_start":
                print_header("PHASE 2 - STRUCTURED EVIDENCE EXTRACTION")
                ui.spinner("Evaluating evidence...")
            elif event_name == "sufficiency_start":
                print_header("EVIDENCE SUFFICIENCY EVALUATION")
            elif event_name == "sufficiency_complete":
                ui.show_evidence_evaluation(payload["sufficiency_results"])
            elif event_name == "evidence_gaps_detected":
                ui.show_evidence_gap(payload["gap_entries"])
            elif event_name == "targeted_research_start":
                ui.show_re_search_launch(payload["query_actions"], additional_sources=0)
                ui.spinner(f"Executing targeted re-search round {payload['attempt']}...")
            elif event_name == "targeted_retrieval_complete":
                stats = payload["retrieval_stats"]
                ui.show_retrieval_complete(stats.get("successes", 0), payload["selected_count"])
                ui.spinner("Re-evaluating evidence...")
            elif event_name == "targeted_sufficiency_evaluated":
                ui.show_evidence_evaluation(payload["sufficiency_results"])
            elif event_name == "summary_start":
                print_header("PHASE 2 - RESEARCH SUMMARY")
                ui.spinner("Synthesizing calibrated research report...")
            elif event_name == "summary_complete":
                print(f"\n{payload['summary']}\n")

        html_report_path = "data/research_report.html"
        orchestrator = ResearchOrchestrator(
            gemini_client=gemini_client,
            tavily_client=tavily_client,
            openalex_provider=openalex_provider,
            semantic_scholar_provider=semantic_scholar_provider,
            model=MODEL,
            max_retries=1,
            on_event=_handle_cli_event,
        )
        pipeline_result = orchestrator.execute_pipeline(
            topic=topic,
            research_plan=research_plan,
            evidence_output_path=EVIDENCE_OUTPUT_FILE,
            synthesize_report=True,
            html_report_path=html_report_path,
            verbose_diagnostics=True,
        )
        evidence_artifact = pipeline_result["evidence_artifact"]
        report_artifact = pipeline_result["report_artifact"]
        sufficiency_results = pipeline_result["sufficiency_results"]

        cal = report_artifact.get("epistemic_calibration", {})
        print(f"\n[Epistemic Calibration] Score: {cal.get('confidence_score')}/1.0 ({cal.get('confidence_tier')})")
        print(f"  {cal.get('tier_description')}")

        from search_agent.tee_logger import TeeLogger
        active_logger = TeeLogger.get_current()
        log_file_path = active_logger.get_log_path() if active_logger else ""

        ui.show_final_screen(
            report_artifact=report_artifact,
            evidence_artifact=evidence_artifact,
            sufficiency_results=sufficiency_results,
            plan_path=PLAN_OUTPUT_FILE,
            evidence_path=EVIDENCE_OUTPUT_FILE,
            report_path="data/research_report.md",
            html_path=html_report_path,
            log_path=log_file_path,
        )

        if sys.stdin.isatty():
            try:
                import webbrowser
                ui.console.print("\n[bold cyan]❯[/] [bold white]Open research report in browser? [Y/n]:[/] ", end="")
                choice = input().strip().lower()
                if choice in ("", "y", "yes"):
                    webbrowser.open("file://" + os.path.abspath(html_report_path))
                    ui.console.print(f"[dim green]Opened {html_report_path} in default browser.[/]\n")
            except (KeyboardInterrupt, EOFError):
                ui.console.print()

    # ======================================================
    # KEYBOARD INTERRUPT
    # ======================================================

    except KeyboardInterrupt:
        if ui is not None:
            ui.stop_spinner()
            ui.console.print("\n\n[dim yellow]Program stopped by user.[/]")
        print("\n\nProgram stopped by user.")
        sys.exit(0)

    # ======================================================
    # ERROR HANDLING
    # ======================================================

    except Exception as error:
        if ui is not None:
            ui.stop_spinner()
            ui.console.print(f"\n[bold red]✖ Error:[/] {error}")
        else:
            try:
                sys.__stdout__.write(f"\nError: {error}\n")
                sys.__stdout__.flush()
            except Exception:
                pass
        print(f"\nError: {error}")
        sys.exit(1)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    is_login = ("login" in sys.argv[1:] or "--login" in sys.argv[1:])
    is_demo = "--demo" in sys.argv
    is_verbose = ("--verbose" in sys.argv or "-v" in sys.argv)
    if is_login:
        from search_agent.auth_manager import run_cli_login_handshake
        creds = run_cli_login_handshake(open_browser=True)
        print(f"Authenticated CLI session for {creds['email']} (org: {creds['org_id']}).")
    elif is_demo:
        from demo import run_demo
        run_demo(verbose=is_verbose)
    else:
        from search_agent.tee_logger import TeeLogger
        with TeeLogger(verbose=is_verbose):
            main()