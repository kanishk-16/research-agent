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

MODEL = "gemini-3.8-flash"


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

        gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip().strip('"').strip("'")

        tavily_key = (os.getenv("TAVILY_API_KEY") or "").strip().strip('"').strip("'")


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

        from main import _DualModeGeminiClient

        gemini_client = _DualModeGeminiClient(gemini_key)


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
        # EXECUTE PHASE 2 via Unified ResearchOrchestrator
        # ==================================================

        from search_agent.orchestrator import ResearchOrchestrator

        def _handle_orchestrator_event(event_name: str, payload: dict):
            if event_name == "discovery_complete":
                _emit(
                    f"Discovery complete: {payload.get('source_count', 0)} unique sources ranked and selected.",
                    type_="ok",
                )
            elif event_name == "extraction_start":
                _emit("Extracting structured evidence from selected sources...")
            elif event_name == "extraction_complete":
                _emit(
                    f"Evidence extraction complete: {payload.get('finding_count', 0)} findings.",
                    type_="ok",
                )
            elif event_name == "sufficiency_start":
                _emit("Evaluating evidence sufficiency...")
            elif event_name == "sufficiency_complete":
                insuff = payload.get("insufficient_questions", [])
                if not insuff:
                    _emit("All questions meet evidence requirements.", type_="ok")
                else:
                    _emit(
                        f"Insufficient evidence for {len(insuff)} question(s). Running targeted re-search..."
                    )
            elif event_name == "targeted_queries_generated":
                _emit(
                    f"{payload.get('question_id', '?')}: {len(payload.get('queries', []))} targeted queries for {payload.get('missing_requirements', [])}"
                )
            elif event_name == "targeted_retrieval_complete":
                _emit(
                    f"Targeted re-search complete. {payload.get('merged_count', 0)} sources merged.",
                    type_="ok",
                )
            elif event_name == "targeted_sufficiency_evaluated":
                insuff = payload.get("insufficient_questions", [])
                if insuff:
                    _emit(
                        f"Targeted re-search finished, but {len(insuff)} question(s) still have evidence gaps."
                    )
                else:
                    _emit("All questions meet evidence requirements after targeted re-search.", type_="ok")
            elif event_name == "summary_start":
                _emit("Generating overall research summary...")
            elif event_name == "summary_complete":
                _emit("Research summary generated.", type_="ok")
            elif event_name == "summary_warning":
                _emit(f"[Warning] Summary generation failed: {payload.get('error', '')}")

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        orchestrator = ResearchOrchestrator(
            gemini_client=gemini_client,
            tavily_client=tavily_client,
            openalex_provider=openalex_provider,
            model=MODEL,
            max_retries=1,
            on_event=_handle_orchestrator_event,
        )
        pipeline_result = orchestrator.execute_pipeline(
            topic=topic,
            research_plan=plan,
            evidence_output_path=str(EVIDENCE_FILE),
            synthesize_report=False,
        )
        _state["evidence"] = pipeline_result["evidence_artifact"]





















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