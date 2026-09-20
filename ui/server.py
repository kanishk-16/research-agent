"""
AARA — Flask backend server

Connects the AARA browser UI to the Phase 1 Planner and
Phase 2 Research pipeline (same flow as main.py).

Usage (from project root):
  .\\venv\\Scripts\\activate
  python ui/server.py

Then open:  http://localhost:5000
"""

import json
import os
import sys
import queue
import threading
from pathlib import Path


# ==========================================================
# PROJECT PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

PLAN_FILE = DATA_DIR / "research_plan.json"
EVIDENCE_FILE = DATA_DIR / "research_evidence.json"

sys.path.insert(0, str(ROOT))


# ==========================================================
# FLASK / ENVIRONMENT
# ==========================================================

from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL = "gemini-3.5-flash-lite"


# ==========================================================
# FLASK APPLICATION
# ==========================================================

app = Flask(
    __name__,
    static_folder=str(UI_DIR),
    static_url_path="",
)

CORS(app)


# ==========================================================
# CURRENT SESSION STATE
# ==========================================================
#
# IMPORTANT:
#
# This dictionary is intentionally NOT populated from old
# research_plan.json / research_evidence.json files.
#
# Restarting server.py therefore creates a clean AARA session.
# ==========================================================

_state = {
    "phase": "idle",
    "topic": "",
    "plan": None,
    "evidence": None,
    "error": None,
}


# ==========================================================
# SSE MESSAGE QUEUE
# ==========================================================

_msg_queue: queue.Queue = queue.Queue()


def _emit(message: str, type_: str = "info", **extra):
    """
    Push a progress event to the browser.
    """

    payload = {
        "message": message,
        "type": type_,
        **extra,
    }

    _msg_queue.put(payload)


def _set_phase(phase: str):
    """
    Update current pipeline phase and notify frontend.
    """

    _state["phase"] = phase

    _emit(
        phase,
        type_="phase",
        phase=phase,
    )


def _clear_message_queue():
    """
    Remove messages belonging to a previous research run.
    """

    while True:

        try:
            _msg_queue.get_nowait()

        except queue.Empty:
            break


# ==========================================================
# FILE SAVING
# ==========================================================

def _save_plan(plan):
    """
    Save current Phase 1 output.

    This file is an artifact only.
    It will NOT be automatically restored next time
    server.py starts.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        PLAN_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            plan,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ==========================================================
# STATIC UI
# ==========================================================

@app.route("/")
def index():
    """
    Serve app.html.
    """

    return send_file(
        UI_DIR / "app.html"
    )


# ==========================================================
# SESSION STATUS
# ==========================================================

@app.route("/api/status")
def api_status():
    """
    Return ONLY current runtime session state.

    Old JSON files are deliberately ignored.
    """

    return jsonify({
        "phase": _state["phase"],
        "topic": _state["topic"],
        "has_plan": _state["plan"] is not None,
        "has_evidence": _state["evidence"] is not None,
        "error": _state["error"],
    })


# ==========================================================
# CURRENT PLAN
# ==========================================================

@app.route("/api/plan")
def api_plan():
    """
    Return current-session research plan.
    """

    plan = _state["plan"]

    if plan is None:
        return jsonify(None), 404

    return jsonify(plan)


# ==========================================================
# CURRENT EVIDENCE
# ==========================================================

@app.route("/api/evidence")
def api_evidence():
    """
    Return current-session Phase 2 evidence.
    """

    evidence = _state["evidence"]

    if evidence is None:
        return jsonify(None), 404

    return jsonify(evidence)


# ==========================================================
# SERVER-SENT EVENTS
# ==========================================================

@app.route("/api/stream")
def api_stream():
    """
    Stream research progress to app.html.
    """

    def event_stream():

        while True:

            try:

                message = _msg_queue.get(
                    timeout=30
                )

                yield (
                    f"data: {json.dumps(message)}\n\n"
                )

                if message.get("phase") in (
                    "done",
                    "error",
                ):
                    break

            except queue.Empty:

                # Keep connection alive.
                yield (
                    'data: {"type":"heartbeat"}\n\n'
                )

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ==========================================================
# START FULL RESEARCH
# ==========================================================

@app.route(
    "/api/research",
    methods=["POST"],
)
def api_research():
    """
    Run Phase 1 + Phase 2.
    """

    body = request.get_json(
        silent=True
    ) or {}

    topic = str(
        body.get("topic", "")
    ).strip()

    if not topic:

        return jsonify({
            "error": "topic is required"
        }), 400


    if _state["phase"] in (
        "planning",
        "researching",
    ):

        return jsonify({
            "error": "Research already running"
        }), 409


    # ------------------------------------------------------
    # NEW RESEARCH = CLEAN CURRENT SESSION
    # ------------------------------------------------------

    _clear_message_queue()

    _state["phase"] = "planning"
    _state["topic"] = topic

    _state["plan"] = None
    _state["evidence"] = None
    _state["error"] = None


    thread = threading.Thread(
        target=_run_pipeline,
        args=(topic, False),
        daemon=True,
    )

    thread.start()


    return jsonify({
        "status": "started",
        "topic": topic,
    })


# ==========================================================
# START PLAN ONLY
# ==========================================================

@app.route(
    "/api/plan_only",
    methods=["POST"],
)
def api_plan_only():
    """
    Run Phase 1 only.
    """

    body = request.get_json(
        silent=True
    ) or {}

    topic = str(
        body.get("topic", "")
    ).strip()


    if not topic:

        return jsonify({
            "error": "topic is required"
        }), 400


    if _state["phase"] in (
        "planning",
        "researching",
    ):

        return jsonify({
            "error": "Research already running"
        }), 409


    # ------------------------------------------------------
    # NEW PLAN = CLEAN CURRENT SESSION
    # ------------------------------------------------------

    _clear_message_queue()

    _state["phase"] = "planning"
    _state["topic"] = topic

    _state["plan"] = None
    _state["evidence"] = None
    _state["error"] = None


    thread = threading.Thread(
        target=_run_pipeline,
        args=(topic, True),
        daemon=True,
    )

    thread.start()


    return jsonify({
        "status": "started",
        "topic": topic,
    })


# ==========================================================
# PIPELINE
# ==========================================================

def _run_pipeline(
    topic: str,
    plan_only: bool,
):
    """
    Execute current AARA Phase 1 / Phase 2 pipeline.
    """

    try:

        # ==================================================
        # API KEYS
        # ==================================================

        gemini_key = os.getenv(
            "GEMINI_API_KEY"
        )

        tavily_key = os.getenv(
            "TAVILY_API_KEY"
        )


        if not gemini_key:

            raise RuntimeError(
                "GEMINI_API_KEY not found in .env"
            )


        if (
            not plan_only
            and not tavily_key
        ):

            raise RuntimeError(
                "TAVILY_API_KEY not found in .env"
            )


        # ==================================================
        # PHASE 1
        # ==================================================

        _set_phase("planning")

        _emit(
            "Phase 1: Generating structured "
            f"research plan for: {topic}"
        )


        # --------------------------------------------------
        # Gemini
        # --------------------------------------------------

        from google import genai

        gemini_client = genai.Client(
            api_key=gemini_key
        )


        # --------------------------------------------------
        # Planner
        # --------------------------------------------------

        from planner.planner import (
            create_research_plan,
        )


        plan = create_research_plan(
            gemini_client,
            MODEL,
            topic,
        )


        _state["plan"] = plan


        question_count = len(
            plan.get(
                "questions",
                [],
            )
        )


        _emit(
            "Research plan created: "
            f"{question_count} questions, "
            f"domain: "
            f"{plan.get('domain', '—')}",
            type_="ok",
        )


        # --------------------------------------------------
        # Save artifact
        # --------------------------------------------------

        _save_plan(plan)


        _emit(
            "Plan saved to "
            "data/research_plan.json"
        )


        # --------------------------------------------------
        # /plan stops here
        # --------------------------------------------------

        if plan_only:

            _emit(
                "Phase 1 complete.",
                type_="ok",
            )

            _set_phase("done")

            return


        # ==================================================
        # PHASE 2
        # ==================================================

        _set_phase("researching")


        _emit(
            "Phase 2: Starting web + "
            "academic source discovery..."
        )


        # ==================================================
        # SEARCH CLIENTS
        # ==================================================

        from tavily import TavilyClient

        tavily_client = TavilyClient(
            api_key=tavily_key
        )


        from search_agent.source_providers import OpenAlexProvider

        openalex_provider = OpenAlexProvider()


        # ==================================================
        # DISPLAY PLANNED SEARCH QUERIES
        # ==================================================

        for question in plan.get(
            "questions",
            [],
        ):

            strategy = question.get(
                "search_strategy",
                {},
            )


            primary_queries = (
                strategy.get(
                    "primary_queries",
                    [],
                )
                or []
            )


            counter_queries = (
                strategy.get(
                    "counter_evidence_queries",
                    [],
                )
                or []
            )


            queries = (
                primary_queries
                + counter_queries
            )


            _emit(
                "",
                type_="queries",
                queries=queries[:4],
                count=len(queries),
                summary=(
                    f"Queries for "
                    f"{question.get('id', '?')}: "
                    f"{question.get('question', '')}"
                ),
            )


        # ==================================================
        # INITIAL RESEARCH
        # ==================================================

        from search_agent.researcher import (
            run_research,
        )


        research_pkg = run_research(
            tavily_client,
            topic,
            plan["questions"],
            gemini_client,
            research_plan=plan,
            openalex_provider=openalex_provider,
        )


        source_count = len(
            research_pkg
        )


        _emit(
            "Discovery complete: "
            f"{source_count} unique sources "
            "ranked and selected.",
            type_="ok",
        )


        # ==================================================
        # EVIDENCE EXTRACTION
        # ==================================================

        _emit(
            "Extracting structured evidence "
            "from selected sources..."
        )


        from search_agent.evidence_extractor import (
            extract_research_evidence,
        )


        selection_bundle = getattr(
            research_pkg,
            "selection_bundle",
            {},
        )


        extraction_bundle = (
            extract_research_evidence(
                gemini_client,
                plan,
                selection_bundle,
            )
        )


        finding_count = len(
            extraction_bundle.get(
                "all_findings",
                [],
            )
        )


        _emit(
            "Evidence extraction complete: "
            f"{finding_count} findings.",
            type_="ok",
        )


        # ==================================================
        # EVIDENCE SUFFICIENCY
        # ==================================================

        _emit(
            "Evaluating evidence sufficiency..."
        )


        from search_agent.evidence_sufficiency import (
            check_all_sufficiency,
            generate_targeted_queries,
        )


        (
            sufficiency_results,
            insufficient_questions,
        ) = check_all_sufficiency(
            plan["questions"],
            extraction_bundle,
            selection_bundle,
        )


        # ==================================================
        # TARGETED RE-SEARCH
        # ==================================================

        if insufficient_questions:

            _emit(
                "Insufficient evidence for "
                f"{len(insufficient_questions)} "
                "question(s). "
                "Running targeted re-search..."
            )


            from search_agent.researcher import (
                run_targeted_research,
            )

            from search_agent.source_ranker import (
                rank_sources,
            )

            from search_agent.source_selector import (
                select_sources,
            )

            from search_agent.content_retriever import (
                retrieve_selected_sources,
            )


            targeted_entries = []


            for (
                question,
                sufficiency_result,
            ) in insufficient_questions:

                missing_requirements = (
                    sufficiency_result.get(
                        "missing_requirements",
                        [],
                    )
                )


                targeted_queries = (
                    generate_targeted_queries(
                        question,
                        missing_requirements,
                    )
                )


                if not targeted_queries:
                    continue


                targeted_entries.append({
                    "question": question,
                    "targeted_queries":
                        targeted_queries,
                })


                _emit(
                    f"{sufficiency_result.get('question_id', '?')}: "
                    f"{len(targeted_queries)} targeted queries "
                    f"for {missing_requirements}"
                )


            # ==================================================
            # EXECUTE TARGETED SEARCH
            # ==================================================

            if targeted_entries:

                targeted_results = (
                    run_targeted_research(
                        tavily_client,
                        topic,
                        targeted_entries,
                        gemini_client=gemini_client,
                        research_plan=plan,
                        openalex_provider=openalex_provider,
                    )
                )


                if targeted_results:

                    from search_agent.sources import (
                        deduplicate_sources,
                    )


                    existing_sources = list(
                        selection_bundle.get(
                            "canonical_sources",
                            list(research_pkg),
                        )
                    )


                    merged_sources = (
                        deduplicate_sources(
                            existing_sources
                            + list(targeted_results)
                        )
                    )


                    # ------------------------------------------
                    # Re-rank
                    # ------------------------------------------

                    merged_ranked = (
                        rank_sources(
                            merged_sources,
                            plan["questions"],
                        )
                    )


                    # ------------------------------------------
                    # Re-select
                    # ------------------------------------------

                    merged_selection = (
                        select_sources(
                            plan["questions"],
                            merged_ranked,
                        )
                    )


                    # ------------------------------------------
                    # Re-retrieve
                    # ------------------------------------------

                    retrieve_selected_sources(
                        tavily_client,
                        merged_selection[
                            "unique_selected_sources"
                        ],
                    )


                    # ------------------------------------------
                    # Re-extract
                    # ------------------------------------------

                    extraction_bundle = (
                        extract_research_evidence(
                            gemini_client,
                            plan,
                            merged_selection,
                        )
                    )


                    selection_bundle = (
                        merged_selection
                    )


                    research_pkg = (
                        merged_ranked
                    )


                    _emit(
                        "Targeted re-search complete. "
                        f"{len(merged_sources)} "
                        "sources merged.",
                        type_="ok",
                    )


                    # ------------------------------------------
                    # Re-evaluate sufficiency
                    # ------------------------------------------

                    (
                        sufficiency_results,
                        insufficient_questions,
                    ) = check_all_sufficiency(
                        plan["questions"],
                        extraction_bundle,
                        selection_bundle,
                    )


                    if insufficient_questions:

                        _emit(
                            "Targeted re-search finished, "
                            f"but {len(insufficient_questions)} "
                            "question(s) still have evidence gaps."
                        )

                    else:

                        _emit(
                            "All questions meet evidence "
                            "requirements after targeted re-search.",
                            type_="ok",
                        )


        else:

            _emit(
                "All questions meet evidence requirements.",
                type_="ok",
            )


        # ==================================================
        # BUILD PHASE 2 OUTPUT
        # ==================================================

        _emit(
            "Generating overall research summary..."
        )

        from search_agent.summary_generator import (
            generate_research_summary,
        )

        selected_sources = getattr(
            research_pkg,
            "unique_selected_sources",
            list(research_pkg),
        )

        try:
            summary = generate_research_summary(
                gemini_client,
                topic,
                selected_sources,
                extraction_bundle=extraction_bundle,
                sufficiency_results=sufficiency_results,
                model=MODEL,
            )
            _emit(
                "Research summary generated.",
                type_="ok",
            )
        except Exception as sum_err:
            summary = ""
            _emit(
                f"[Warning] Summary generation failed: {sum_err}"
            )

        _emit(
            "Building research evidence artifact..."
        )

        from search_agent.evidence_output import (
            build_research_evidence_output,
            save_research_evidence,
        )

        evidence_artifact = (
            build_research_evidence_output(
                topic,
                research_pkg,
                extraction_bundle,
                summary=summary,
                sufficiency_results=sufficiency_results,
            )
        )


        # ==================================================
        # SAVE EVIDENCE
        # ==================================================

        DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )


        save_research_evidence(
            evidence_artifact,
            str(EVIDENCE_FILE),
        )


        _state["evidence"] = (
            evidence_artifact
        )


        # ==================================================
        # FINISHED
        # ==================================================

        _emit(
            "Pipeline complete. "
            "research_evidence.json saved.",
            type_="ok",
        )


        _set_phase("done")


    # ======================================================
    # ERROR
    # ======================================================

    except Exception as exc:

        error_message = str(exc)


        _state["error"] = (
            error_message
        )


        _emit(
            "Pipeline failed: "
            f"{error_message}",
            type_="error",
        )


        _set_phase("error")


        import traceback

        traceback.print_exc()


# ==========================================================
# START SERVER
# ==========================================================

if __name__ == "__main__":

    print()
    print("=" * 64)
    print(" AARA — Autonomous Agentic Research Agent")
    print("=" * 64)

    print()
    print(" Fresh AARA session started.")
    print(" Previous research artifacts will NOT be restored.")
    print()

    print(
        " Server: http://localhost:5000"
    )

    print()
    print("=" * 64)
    print()


    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )