"""
cli/theme.py

Design system and visual tokens for Autonomous Research Agent (ARA).
Defines restrained colors, Unicode glyphs, typography styles, and formatting helpers
inspired by modern developer CLI tools.
"""

from rich.theme import Theme
from rich.style import Style
from rich.text import Text

# ----------------------------------------------------------------------
# UNICODE GLYPHS & SYMBOLS
# ----------------------------------------------------------------------
SYM_PROMPT = "❯"
SYM_BULLET = "●"
SYM_CHECK = "✓"
SYM_HALF = "◐"
SYM_CROSS = "×"
SYM_DIAMOND = "◇"
SYM_REFRESH = "↻"
SYM_WARNING = "⚠"
SYM_BRANCH = "├─"
SYM_LAST = "└─"
SYM_VERT = "│"
SYM_BAR_FILL = "█"
SYM_BAR_EMPTY = "░"
SYM_ARROW = "→"
SYM_DOT = "·"

# ----------------------------------------------------------------------
# RICH THEME TOKENS
# ----------------------------------------------------------------------
CLI_THEME = Theme({
    "ara.brand": "bold cyan",
    "ara.logo": "bold cyan",
    "ara.tagline": "dim italic",
    "ara.rule": "dim #475569",
    "ara.subtle": "#64748b",
    "ara.dim": "dim",
    "ara.text": "white",
    "ara.bold": "bold white",
    "ara.prompt": "bold cyan",
    "ara.bullet": "bold cyan",
    "ara.diamond": "bold yellow",
    "ara.refresh": "bold cyan",
    "ara.success": "bold green",
    "ara.warning": "bold yellow",
    "ara.error": "bold red",
    "ara.tree": "dim #64748b",
    "ara.metric": "bold white",
    "ara.label": "dim",
    "ara.path": "underline cyan",
})


def format_confidence_bar(score: float, tier: str, width: int = 20) -> Text:
    """Format an epistemic confidence progress bar with tier and score."""
    clamped = max(0.0, min(1.0, float(score or 0.0)))
    filled_len = int(round(clamped * width))
    empty_len = max(0, width - filled_len)
    tier_label = str(tier or "UNKNOWN").upper()

    t = Text()
    t.append(f"{tier_label:<10}  ", style="bold")
    t.append(SYM_BAR_FILL * filled_len, style="green")
    t.append(SYM_BAR_EMPTY * empty_len, style="dim")
    t.append(f"  {clamped:.2f}", style="bold white")
    return t


def format_status_indicator(sufficient: bool) -> str:
    """Format a clean sufficient/insufficient indicator."""
    if sufficient:
        return f"[bold green]{SYM_CHECK} sufficient[/]"
    return f"[bold yellow]{SYM_HALF} insufficient[/]"
