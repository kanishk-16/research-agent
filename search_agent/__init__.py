"""Phase 2 research-agent package."""

from .researcher import run_research
from .source_selector import select_sources
from .content_retriever import retrieve_selected_sources
from .evidence_extractor import extract_research_evidence
from .evidence_output import build_research_evidence_output, save_research_evidence
from .source_providers import (
    SourceProvider,
    OpenAlexProvider,
    register_provider,
    get_provider,
    search_academic,
)

__all__ = [
    "run_research",
    "select_sources",
    "retrieve_selected_sources",
    "extract_research_evidence",
    "build_research_evidence_output",
    "save_research_evidence",
    "SourceProvider",
    "OpenAlexProvider",
    "register_provider",
    "get_provider",
    "search_academic",
]
