"""
cli/components.py

Terminal UI visual components for Autonomous Research Agent (ARA).
Constructs ASCII art branding, status tables, gap alerts, tree nodes,
and the final research synthesis panel.
"""

from typing import List, Dict, Any, Optional
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.box import ROUNDED, SIMPLE, HORIZONTALS

from .theme import (
    SYM_PROMPT,
    SYM_BULLET,
    SYM_CHECK,
    SYM_HALF,
    SYM_CROSS,
    SYM_DIAMOND,
    SYM_REFRESH,
    SYM_WARNING,
    SYM_BRANCH,
    SYM_LAST,
    SYM_VERT,
    SYM_ARROW,
    SYM_DOT,
    format_confidence_bar,
)

# ----------------------------------------------------------------------
# ASCII BRANDING
# ----------------------------------------------------------------------
ARA_ASCII_LOGO = r"""
     █████╗  ██████╗   █████╗ 
    ██╔══██╗ ██╔══██╗ ██╔══██╗
    ███████║ ██████╔╝ ███████║
    ██╔══██║ ██╔══██╗ ██╔══██║
    ██║  ██║ ██║  ██║ ██║  ██║
    ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝"""


def render_banner() -> Text:
    """Render the main ARA ASCII logo and title subtitle."""
    text = Text()
    text.append(ARA_ASCII_LOGO, style="bold cyan")
    text.append("\n\n       AUTONOMOUS RESEARCH AGENT\n", style="bold white")
    text.append("   Evidence-driven research from question to conclusion\n", style="dim italic")
    return text


def render_startup_status(
    openalex_active: bool = True,
    semantic_scholar_active: bool = True,
    tavily_active: bool = True,
    width: int = 60
) -> Text:
    """Render the startup provider and mode status box."""
    t = Text()
    rule_line = "─" * width

    academic_parts = []
    if openalex_active:
        academic_parts.append("OpenAlex")
    if semantic_scholar_active:
        academic_parts.append("Semantic Scholar")
    academic_str = f" {SYM_DOT} ".join(academic_parts) if academic_parts else "None"

    web_str = "Tavily" if tavily_active else "Offline"

    t.append(f"{rule_line}\n\n", style="dim")
    t.append("  Academic Search     ", style="dim")
    t.append(f"{academic_str}\n", style="white")
    t.append("  Web Research        ", style="dim")
    t.append(f"{web_str}\n", style="white")
    t.append("  Evidence Engine     ", style="dim")
    t.append("Active\n", style="bold green")
    t.append("  Research Mode       ", style="dim")
    t.append("Autonomous\n\n", style="cyan")
    t.append(f"{rule_line}\n", style="dim")
    return t


def render_evidence_gap_alert(insufficient_questions: List[Dict[str, Any]]) -> Text:
    """Render the evidence gap detected card."""
    t = Text()
    t.append(f"{SYM_DIAMOND} Evidence gap detected\n\n", style="bold yellow")
    for q_data in insufficient_questions:
        qid = q_data.get("question_id", "")
        missing = q_data.get("missing_requirements", [])
        for m in missing:
            m_label = m.replace("_", " ")
            t.append(f"  {qid} needs additional {m_label}\n", style="dim")
    return t


def render_re_search_launch(actions: List[str], extra_source_count: Optional[int] = None) -> Text:
    """Render the autonomous re-search launch card."""
    t = Text()
    t.append(f"\n{SYM_REFRESH} Launching targeted research...\n\n", style="bold cyan")
    all_actions = list(actions)
    if extra_source_count is not None and extra_source_count > 0:
        all_actions.append(f"retrieving {extra_source_count} additional sources")

    for i, action in enumerate(all_actions):
        is_last = (i == len(all_actions) - 1)
        branch = SYM_LAST if is_last else SYM_BRANCH
        t.append(f"  {branch} {action}\n", style="dim")
    return t


def render_provider_warning(provider_name: str, issue: str, fallback: str = "") -> Text:
    """Render a clean provider status alert."""
    t = Text()
    t.append(f"{SYM_WARNING} {provider_name} {issue}\n", style="bold yellow")
    if fallback:
        t.append(f"  {fallback}\n", style="dim")
    return t


def render_evidence_table(sufficiency_results: List[Dict[str, Any]]) -> Text:
    """Render compact question evidence status rows."""
    t = Text()
    for sr in sufficiency_results:
        qid = sr.get("question_id", "")
        is_suff = sr.get("sufficient", False)
        f_count = sr.get("finding_count", 0)

        sym = SYM_CHECK if is_suff else SYM_HALF
        color = "green" if is_suff else "yellow"
        status_word = "sufficient" if is_suff else "insufficient"

        # Format: Q1  ✓ sufficient       10 findings
        findings_str = f"{f_count} findings" if f_count != 1 else "1 finding"
        t.append(f"  {qid:<4} ", style="bold white")
        t.append(f"{sym} {status_word:<13} ", style=f"bold {color}")
        t.append(f"{findings_str}\n", style="dim")
    return t


def render_final_screen(
    sufficiency_results: List[Dict[str, Any]],
    confidence_score: float,
    confidence_tier: str,
    total_sources: int,
    total_quant: int,
    total_counter: int,
    research_rounds: int,
    plan_path: str = "data/research_plan.json",
    evidence_path: str = "data/research_evidence.json",
    report_path: str = "data/research_report.md",
    html_path: str = "data/research_report.html",
    log_path: str = "",
    width: int = 60
) -> RenderableType:
    """Render the polished completion screen and summary panel."""
    # Header Panel
    header_content = Text(f" {SYM_CHECK} Research complete", style="bold green")
    header_panel = Panel(
        header_content,
        box=ROUNDED,
        border_style="green",
        expand=False,
        padding=(0, 2)
    )

    t = Text()
    rule_line = "─" * width

    # Evidence section
    t.append("\n  Evidence\n\n", style="bold white")
    for sr in sufficiency_results:
        qid = sr.get("question_id", "")
        is_suff = sr.get("sufficient", False)
        f_count = sr.get("finding_count", 0)

        sym = SYM_CHECK if is_suff else SYM_HALF
        color = "green" if is_suff else "yellow"
        status_word = "Sufficient" if is_suff else "Insufficient"
        findings_str = f"{f_count:>2} findings"

        t.append(f"  {qid:<4} ", style="bold white")
        t.append(f"{sym} {status_word:<14} ", style=f"bold {color}")
        t.append(f"{findings_str}\n", style="dim")

    # Epistemic Confidence Bar
    t.append("\n  Epistemic confidence\n  ", style="bold white")
    t.append_text(format_confidence_bar(confidence_score, confidence_tier, width=20))
    t.append("\n\n")

    # Metrics
    t.append("  Sources analyzed       ", style="dim")
    t.append(f"{total_sources:>2}\n", style="bold white")
    t.append("  Quantitative findings  ", style="dim")
    t.append(f"{total_quant:>2}\n", style="bold white")
    t.append("  Counter-evidence       ", style="dim")
    t.append(f"{total_counter:>2}\n", style="bold white")
    t.append("  Research rounds        ", style="dim")
    t.append(f"{research_rounds:>2}\n\n", style="bold white")

    # Artifact paths
    t.append(f"{rule_line}\n\n", style="dim")
    if html_path:
        t.append("  Report (HTML) ", style="bold green")
        t.append(f"{html_path}\n", style="bold white")
    t.append("  Report (MD)   ", style="dim")
    t.append(f"{report_path}\n", style="cyan")
    t.append("  Evidence      ", style="dim")
    t.append(f"{evidence_path}\n", style="cyan")
    t.append("  Plan          ", style="dim")
    t.append(f"{plan_path}\n", style="cyan")
    if log_path:
        t.append("  Log           ", style="dim")
        t.append(f"{log_path}\n", style="dim cyan")
    t.append(f"\n{rule_line}\n", style="dim")

    # Group components
    from rich.console import Group
    return Group(header_panel, t)
