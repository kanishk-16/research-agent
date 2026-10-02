"""
cli/progress.py

Live stage progress and transitions for Autonomous Research Agent (ARA).
Coordinates subtle spinners during asynchronous tasks and replaces them with
clean, permanent hierarchical tree outputs upon completion.
"""

from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.text import Text
from rich.status import Status

from .theme import (
    SYM_BULLET,
    SYM_CHECK,
    SYM_HALF,
    SYM_DIAMOND,
    SYM_REFRESH,
    SYM_BRANCH,
    SYM_LAST,
    SYM_BAR_FILL,
    SYM_BAR_EMPTY,
    SYM_ARROW,
    SYM_DOT,
)


class ResearchProgressManager:
    """Manages live spinners and compact hierarchical terminal stages."""

    def __init__(self, console: Console):
        self.console = console
        self._active_status: Optional[Status] = None

    def start_spinner(self, message: str) -> None:
        """Start a subtle terminal spinner for an active background task."""
        self.stop_spinner()
        self._active_status = self.console.status(
            f"[dim]{message}[/]",
            spinner="dots",
            spinner_style="cyan"
        )
        if hasattr(self._active_status, "_live"):
            self._active_status._live._redirect_stdout = False
            self._active_status._live._redirect_stderr = False
        self._active_status.start()

    def update_spinner(self, message: str) -> None:
        """Update the text of the currently active spinner."""
        if self._active_status:
            self._active_status.update(f"[dim]{message}[/]")

    def stop_spinner(self) -> None:
        """Stop and clear the active spinner."""
        if self._active_status:
            self._active_status.stop()
            self._active_status = None

    # ------------------------------------------------------------------
    # STAGE 1: PLANNING
    # ------------------------------------------------------------------
    def complete_planning(self, question_count: int, has_counter: bool = True) -> None:
        """Render the completed planning stage tree."""
        self.stop_spinner()
        t = Text()
        t.append(f"\n{SYM_BULLET} Planning research\n\n", style="bold cyan")
        t.append(f"  {SYM_BRANCH} Generated {question_count} research questions\n", style="dim")
        t.append(f"  {SYM_BRANCH} Defined evidence requirements\n", style="dim")
        t.append(f"  {SYM_LAST} Created counter-evidence strategy\n", style="dim")
        self.console.print(t)

    # ------------------------------------------------------------------
    # STAGE 2: SEARCHING & DISCOVERY
    # ------------------------------------------------------------------
    def complete_search(
        self,
        providers: List[str],
        candidate_count: int,
        unique_count: int
    ) -> None:
        """Render the completed search stage tree."""
        self.stop_spinner()
        t = Text()
        t.append(f"\n{SYM_BULLET} Searching\n\n", style="bold cyan")
        for i, provider in enumerate(providers):
            is_last = (i == len(providers) - 1)
            branch = SYM_LAST if is_last else SYM_BRANCH
            t.append(f"  {branch} {provider}\n", style="dim")

        t.append(f"\n  {candidate_count} candidates {SYM_ARROW} ", style="dim")
        t.append(f"{unique_count} unique sources\n", style="bold white")
        self.console.print(t)

    # ------------------------------------------------------------------
    # STAGE 3: RANKING
    # ------------------------------------------------------------------
    def complete_ranking(self, selected_count: int) -> None:
        """Render the completed ranking stage tree."""
        self.stop_spinner()
        t = Text()
        t.append(f"\n{SYM_BULLET} Ranking sources\n\n", style="bold cyan")
        t.append(f"  {SYM_LAST} {selected_count} sources selected for deep retrieval\n", style="dim")
        self.console.print(t)

    # ------------------------------------------------------------------
    # STAGE 4: RETRIEVAL
    # ------------------------------------------------------------------
    def complete_retrieval(self, retrieved: int, total: int, width: int = 20) -> None:
        """Render the completed retrieval progress bar."""
        self.stop_spinner()
        total_safe = max(1, total)
        ratio = min(1.0, retrieved / total_safe)
        filled = int(round(ratio * width))
        empty = max(0, width - filled)

        t = Text()
        t.append(f"\n{SYM_BULLET} Retrieving evidence\n\n", style="bold cyan")
        t.append(f"  {SYM_BAR_FILL * filled}", style="cyan")
        t.append(f"{SYM_BAR_EMPTY * empty}", style="dim")
        t.append(f"  {retrieved} / {total}\n", style="bold white")
        self.console.print(t)

    # ------------------------------------------------------------------
    # STAGE 5: EVIDENCE EVALUATION
    # ------------------------------------------------------------------
    def complete_evidence_evaluation(self, sufficiency_results: List[Dict[str, Any]]) -> None:
        """Render the question-level evidence evaluation status."""
        self.stop_spinner()
        t = Text()
        t.append(f"\n{SYM_BULLET} Evaluating evidence\n\n", style="bold cyan")
        for sr in sufficiency_results:
            qid = sr.get("question_id", "")
            is_suff = sr.get("sufficient", False)
            f_count = sr.get("finding_count", 0)

            sym = SYM_CHECK if is_suff else SYM_HALF
            color = "green" if is_suff else "yellow"
            status_word = "sufficient" if is_suff else "insufficient"
            findings_str = f"{f_count} findings" if f_count != 1 else "1 finding"

            t.append(f"  {qid:<4} ", style="bold white")
            t.append(f"{sym} {status_word:<13} ", style=f"bold {color}")
            t.append(f"{findings_str}\n", style="dim")

        self.console.print(t)

    # ------------------------------------------------------------------
    # TARGETED RE-SEARCH
    # ------------------------------------------------------------------
    def show_evidence_gap(self, insufficient_entries: List[Dict[str, Any]]) -> None:
        """Render detected evidence gaps."""
        self.stop_spinner()
        t = Text()
        t.append(f"\n{SYM_DIAMOND} Evidence gap detected\n\n", style="bold yellow")
        for entry in insufficient_entries:
            qid = entry.get("question_id", "")
            missing = entry.get("missing_requirements", [])
            for m in missing:
                m_label = m.replace("_", " ")
                t.append(f"  {qid} needs additional {m_label}\n", style="dim")
        self.console.print(t)

    def show_re_search_launch(self, query_actions: List[str], additional_sources: int) -> None:
        """Render targeted re-search launch actions."""
        self.stop_spinner()
        t = Text()
        t.append(f"\n{SYM_REFRESH} Launching targeted research...\n\n", style="bold cyan")
        all_actions = list(query_actions)
        if additional_sources > 0:
            all_actions.append(f"retrieving {additional_sources} additional sources")

        for i, action in enumerate(all_actions):
            is_last = (i == len(all_actions) - 1)
            branch = SYM_LAST if is_last else SYM_BRANCH
            t.append(f"  {branch} {action}\n", style="dim")
        self.console.print(t)
