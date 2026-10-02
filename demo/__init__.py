"""
demo package for Autonomous Research Agent.
"""

from .demo_runner import run_demo, NetworkGuardViolation
from .demo_data import (
    DEMO_TOPIC,
    DEMO_PLAN,
    DEMO_ROUND1_SUFFICIENCY,
    DEMO_ROUND2_SUFFICIENCY,
    DEMO_EVIDENCE_ARTIFACT,
    DEMO_REPORT_ARTIFACT,
)

__all__ = [
    "run_demo",
    "NetworkGuardViolation",
    "DEMO_TOPIC",
    "DEMO_PLAN",
    "DEMO_ROUND1_SUFFICIENCY",
    "DEMO_ROUND2_SUFFICIENCY",
    "DEMO_EVIDENCE_ARTIFACT",
    "DEMO_REPORT_ARTIFACT",
]
