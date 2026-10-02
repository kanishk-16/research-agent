"""
cli/console.py

Main Console and UI controller for Autonomous Research Agent (ARA).
Provides the primary interface between the research engine and the terminal UI.
"""

import os
import sys
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.text import Text

from .theme import CLI_THEME, SYM_PROMPT, SYM_WARNING
from .components import (
    render_banner,
    render_startup_status,
    render_provider_warning,
    render_final_screen,
)
from .progress import ResearchProgressManager


class ARAConsole:
    """The central UI Controller for ARA."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        # Use sys.__stdout__ to write directly to terminal even when stdout is tee'd
        target_stream = getattr(sys, "__stdout__", sys.stdout)
        if hasattr(target_stream, "reconfigure"):
            try:
                target_stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        self.console = Console(
            file=target_stream,
            theme=CLI_THEME,
            highlight=False,
            markup=True
        )
        self.progress = ResearchProgressManager(self.console)
        self._research_rounds = 1
        self._warned_providers = set()

    def show_banner(self) -> None:
        """Render the ARA startup banner and visual identity."""
        self.console.print(render_banner())

    def show_startup(
        self,
        openalex_active: bool = True,
        semantic_scholar_active: bool = True,
        tavily_active: bool = True
    ) -> None:
        """Render the startup providers and operating mode."""
        self.console.print(
            render_startup_status(
                openalex_active=openalex_active,
                semantic_scholar_active=semantic_scholar_active,
                tavily_active=tavily_active
            )
        )

    def prompt_question(self, default_value: str = "") -> str:
        """Prompt the user for their research question using a modern CLI prompt."""
        if default_value:
            self.console.print(f"[bold cyan]{SYM_PROMPT}[/] [bold white]What would you like to research?[/]")
            self.console.print(f"  [dim cyan]{default_value}[/]\n")
            return default_value

        self.console.print(f"\n[bold cyan]{SYM_PROMPT}[/] [bold white]What would you like to research?[/]")
        try:
            user_input = input("  ❯ ").strip()
            self.console.print()
            return user_input
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[dim]Research session cancelled by user.[/]")
            sys.exit(0)

    def prompt_clarification(self, reason: str, questions: List[str]) -> str:
        """Prompt user to resolve an ambiguous or underspecified topic."""
        self.progress.stop_spinner()
        self.console.print(f"\n[bold yellow]{SYM_WARNING} Ambiguity Detected[/]")
        self.console.print(f"  [dim]{reason}[/]\n")
        self.console.print("[dim]Please clarify:[/]")
        for q in questions:
            self.console.print(f"  [dim]- {q}[/]")

        self.console.print(f"\n[bold cyan]{SYM_PROMPT}[/] [bold white]Enter clarified research question (or press Enter to cancel):[/]")
        try:
            clarified = input("  ❯ ").strip()
            self.console.print()
            return clarified
        except (KeyboardInterrupt, EOFError):
            return ""

    def spinner(self, text: str) -> None:
        """Start or update active stage spinner."""
        self.progress.start_spinner(text)

    def stop_spinner(self) -> None:
        """Stop active stage spinner."""
        self.progress.stop_spinner()

    def show_planning_complete(self, research_plan: Dict[str, Any]) -> None:
        """Display completed planning stage."""
        questions = research_plan.get("questions", [])
        has_counter = any(q.get("requires_counter_evidence", False) for q in questions)
        self.progress.complete_planning(len(questions), has_counter=has_counter)

    def show_search_complete(
        self,
        candidate_count: int,
        unique_count: int,
        providers: Optional[List[str]] = None
    ) -> None:
        """Display completed discovery search stage."""
        if providers is None:
            providers = ["Tavily", "OpenAlex", "Semantic Scholar"]
        self.progress.complete_search(providers, candidate_count, unique_count)

    def show_ranking_complete(self, selected_count: int) -> None:
        """Display completed ranking stage."""
        self.progress.complete_ranking(selected_count)

    def show_retrieval_complete(self, successes: int, total: int) -> None:
        """Display completed full-text retrieval progress."""
        self.progress.complete_retrieval(successes, total)

    def show_evidence_evaluation(self, sufficiency_results: List[Dict[str, Any]]) -> None:
        """Display question evidence sufficiency status."""
        self.progress.complete_evidence_evaluation(sufficiency_results)

    def show_evidence_gap(self, insufficient_entries: List[Dict[str, Any]]) -> None:
        """Display detected evidence gaps before targeted re-search."""
        self.progress.show_evidence_gap(insufficient_entries)

    def show_re_search_launch(
        self,
        actions: List[str],
        additional_sources: int = 0
    ) -> None:
        """Display autonomous re-search launch card."""
        self._research_rounds += 1
        self.progress.show_re_search_launch(actions, additional_sources)

    def show_provider_warning(
        self,
        provider_name: str,
        issue: str,
        fallback: str = ""
    ) -> None:
        """Display clean provider issue warning (debounced to avoid repetitive alert spam)."""
        cache_key = f"{provider_name}:{issue}"
        if cache_key in self._warned_providers:
            return
        self._warned_providers.add(cache_key)
        self.progress.stop_spinner()
        self.console.print(render_provider_warning(provider_name, issue, fallback))

    def show_final_screen(
        self,
        report_artifact: Dict[str, Any],
        evidence_artifact: Dict[str, Any],
        sufficiency_results: List[Dict[str, Any]],
        plan_path: str = "data/research_plan.json",
        evidence_path: str = "data/research_evidence.json",
        report_path: str = "data/research_report.md",
        html_path: str = "data/research_report.html",
        log_path: str = "",
    ) -> None:
        """Render the complete final screen and metrics summary."""
        self.progress.stop_spinner()

        cal = report_artifact.get("epistemic_calibration", {})
        confidence_score = float(cal.get("confidence_score") or 0.0)
        confidence_tier = str(cal.get("confidence_tier") or "UNKNOWN")

        stats = evidence_artifact.get("statistics", {})
        total_sources = stats.get("selected_sources") or len(evidence_artifact.get("sources", []))

        total_quant = stats.get("quantitative_findings")
        if total_quant is None:
            total_quant = sum(int(sr.get("quantitative_count", 0)) for sr in sufficiency_results)

        total_counter = stats.get("counter_findings")
        if total_counter is None:
            total_counter = sum(int(sr.get("counter_count", 0)) for sr in sufficiency_results)

        # Fallback to direct inspection of questions if both zero
        if total_quant == 0 and total_counter == 0:
            for q in evidence_artifact.get("questions", []):
                for f in q.get("findings", []):
                    if f.get("quantitative_evidence"):
                        total_quant += len(f["quantitative_evidence"])
                    if f.get("stance") == "counter":
                        total_counter += 1

        self.console.print()
        self.console.print(
            render_final_screen(
                sufficiency_results=sufficiency_results,
                confidence_score=confidence_score,
                confidence_tier=confidence_tier,
                total_sources=total_sources,
                total_quant=total_quant,
                total_counter=total_counter,
                research_rounds=self._research_rounds,
                plan_path=plan_path,
                evidence_path=evidence_path,
                report_path=report_path,
                html_path=html_path,
                log_path=log_path
            )
        )
