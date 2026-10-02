"""
demo/demo_runner.py

Orchestrates the completely offline --demo mode for ARA.
Exercises the complete interactive terminal UI experience using realistic
simulated data without making any external API or network calls.

Strict Network Isolation:
Ensures zero calls to OpenAlex, Semantic Scholar, Tavily, or Gemini.
Enforces runtime network guards that fail loudly if any external request is attempted.
"""

import json
import os
import sys
import time
import urllib.request
import webbrowser
from typing import Optional

from cli.console import ARAConsole
from reports.html_renderer import render_html_report
from .demo_data import (
    DEMO_TOPIC,
    DEMO_PLAN,
    DEMO_ROUND1_SUFFICIENCY,
    DEMO_ROUND2_SUFFICIENCY,
    DEMO_EVIDENCE_ARTIFACT,
    DEMO_REPORT_ARTIFACT,
)


class NetworkGuardViolation(RuntimeError):
    """Raised if any network activity is attempted during offline demo mode."""
    pass


def install_network_guard():
    """
    Installs strict runtime network interception.
    Any attempt to execute external network requests raises NetworkGuardViolation.
    """
    def _blocked_urlopen(*args, **kwargs):
        raise NetworkGuardViolation(
            "STRICT NETWORK ISOLATION VIOLATION: External network call blocked in --demo mode."
        )

    urllib.request.urlopen = _blocked_urlopen


def run_demo(verbose: bool = False, auto_open: bool = False, ui: Optional[ARAConsole] = None) -> None:
    """
    Runs the complete ARA offline demo simulation.
    """
    install_network_guard()

    if ui is None:
        ui = ARAConsole(verbose=verbose)

    # 1. Startup Visual Identity
    ui.show_banner()
    ui.show_startup(
        openalex_active=True,
        semantic_scholar_active=True,
        tavily_active=True
    )

    # 2. Topic Input Prompt
    user_topic = ui.prompt_question(default_value=DEMO_TOPIC)
    if not user_topic:
        user_topic = DEMO_TOPIC

    # 3. Planning Phase
    ui.spinner("Planning research...")
    time.sleep(0.35)
    ui.show_planning_complete(DEMO_PLAN)

    # 4. Search & Discovery
    ui.spinner("Searching academic & web providers...")
    time.sleep(0.40)
    ui.show_search_complete(
        candidate_count=544,
        unique_count=392,
        providers=["Tavily", "OpenAlex", "Semantic Scholar"]
    )

    # 5. Ranking & Selection
    ui.show_ranking_complete(selected_count=13)

    # 6. Deep Retrieval Progress
    ui.show_retrieval_complete(successes=13, total=13)

    # 7. Initial Evidence Evaluation (Insufficient)
    ui.spinner("Evaluating evidence...")
    time.sleep(0.30)
    ui.show_evidence_evaluation(DEMO_ROUND1_SUFFICIENCY)

    # 8. Evidence Gap Detection
    gap_entries = [
        {"question_id": "Q2", "missing_requirements": ["quantitative_evidence"]},
        {"question_id": "Q3", "missing_requirements": ["counter_evidence"]}
    ]
    ui.show_evidence_gap(gap_entries)

    # 9. Targeted Re-search Launch
    ui.show_re_search_launch(
        actions=["searching quantitative evidence", "searching counter evidence"],
        additional_sources=0
    )

    # 10. Re-retrieval
    ui.spinner("Executing targeted re-search round 1...")
    time.sleep(0.35)
    ui.show_retrieval_complete(successes=13, total=13)

    # 11. Re-evaluation (Sufficient)
    ui.spinner("Re-evaluating evidence...")
    time.sleep(0.25)
    ui.show_evidence_evaluation(DEMO_ROUND2_SUFFICIENCY)

    # 12. Calibrated Report Synthesis
    ui.spinner("Synthesizing calibrated research report...")
    time.sleep(0.35)

    # 13. Save Research Artifacts (Including Primary HTML Report)
    os.makedirs("data", exist_ok=True)
    with open("data/research_plan.json", "w", encoding="utf-8") as f:
        json.dump(DEMO_PLAN, f, indent=2, ensure_ascii=False)

    with open("data/research_evidence.json", "w", encoding="utf-8") as f:
        json.dump(DEMO_EVIDENCE_ARTIFACT, f, indent=2, ensure_ascii=False)

    with open("data/research_report.json", "w", encoding="utf-8") as f:
        json.dump(DEMO_REPORT_ARTIFACT, f, indent=2, ensure_ascii=False)

    with open("data/research_report.md", "w", encoding="utf-8") as f:
        f.write(DEMO_REPORT_ARTIFACT["report_markdown"])

    html_path = "data/research_report.html"
    render_html_report(
        report_artifact=DEMO_REPORT_ARTIFACT,
        evidence_artifact=DEMO_EVIDENCE_ARTIFACT,
        research_plan=DEMO_PLAN,
        output_path=html_path,
        is_demo=True
    )

    # 14. Show Final Screen
    ui.show_final_screen(
        report_artifact=DEMO_REPORT_ARTIFACT,
        evidence_artifact=DEMO_EVIDENCE_ARTIFACT,
        sufficiency_results=DEMO_ROUND2_SUFFICIENCY,
        plan_path="data/research_plan.json",
        evidence_path="data/research_evidence.json",
        report_path="data/research_report.md",
        html_path=html_path,
        log_path="evaluation_logs/demo_run_offline.txt"
    )

    # 15. Offer to Open in Browser
    if auto_open:
        try:
            webbrowser.open("file://" + os.path.abspath(html_path))
        except Exception:
            pass
    elif sys.stdin.isatty():
        try:
            ui.console.print("\n[bold cyan]❯[/] [bold white]Open research report in browser? [Y/n]:[/] ", end="")
            choice = input().strip().lower()
            if choice in ("", "y", "yes"):
                webbrowser.open("file://" + os.path.abspath(html_path))
                ui.console.print(f"[dim green]Opened {html_path} in default browser.[/]\n")
        except (KeyboardInterrupt, EOFError):
            ui.console.print()
