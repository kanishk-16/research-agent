"""
reports package for Autonomous Research Agent.
"""

from .html_renderer import (
    render_html_report,
    select_canonical_url,
    build_citation_map,
    format_authors,
)

__all__ = [
    "render_html_report",
    "select_canonical_url",
    "build_citation_map",
    "format_authors",
]
