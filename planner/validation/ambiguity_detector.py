"""
Ambiguity and Clarification Detector for Research Topics.

Prevents the Planner from inventing an arbitrary research direction when given
an ambiguous, placeholder, or underspecified research query.
"""

import re
from typing import Tuple, Optional, List


PLACEHOLDER_TOPIC_PATTERNS = [
    r"controversial\s+scientific\s+claim\b",
    r"a\s+certain\s+(?:claim|phenomenon|technique|method)\b",
    r"the\s+(?:proposed|new)\s+(?:technique|approach|method|system)\s+vs\s+the\s+baseline\b",
    r"^what\s+evidence\s+exists\s+regarding\s+(?:a|the)\s+(?:claim|topic|idea)\??$",
    r"^(?:tell\s+me\s+about\s+)?research\b(?:\s+topics?)?\.?$",
    r"^(?:is\s+it\s+true\s+or\s+false|does\s+it\s+work)\??$",
    r"^evaluate\s+(?:the\s+)?(?:hypothesis|claim)\??$",
    r"^(?:recent\s+)?scientific\s+studies\??$",
]

ABSTRACT_PLACEHOLDER_SUBJECTS = {
    "controversial claim",
    "scientific claim",
    "a claim",
    "some method",
    "the method",
    "the hypothesis",
    "the system",
    "certain technique",
    "recent study",
    "unnamed topic",
}


def detect_topic_ambiguity(topic: str) -> Tuple[bool, Optional[str], List[str]]:
    """
    Detect whether the user's research topic is ambiguous or a generic placeholder.
    Returns:
      (is_ambiguous, reason_description, suggested_clarifications)
    """
    clean_topic = topic.strip().lower()
    if not clean_topic:
        return True, "Research topic is empty.", [
            "Please provide a specific scientific hypothesis or research topic."
        ]

    # 1. Check direct placeholder patterns
    for p in PLACEHOLDER_TOPIC_PATTERNS:
        if re.search(p, clean_topic):
            return True, (
                "The topic refers to a generic placeholder rather than a concrete entity, "
                "phenomenon, or hypothesis."
            ), [
                "Which specific scientific or technological claim would you like to investigate?",
                "What specific models, architectures, or empirical domains should be compared?",
                "Are you evaluating a specific benchmark, dataset, or experimental metric?"
            ]

    # 2. Check for abstract placeholder subjects
    for subj in ABSTRACT_PLACEHOLDER_SUBJECTS:
        if subj in clean_topic and len(clean_topic.split()) < 8:
            return True, (
                f"The topic contains generic phrase '{subj}' without specifying the actual subject."
            ), [
                "Please specify the subject or domain (e.g., 'Multi-agent LLM debate on GSM8K reasoning').",
                "What are the baseline and experimental techniques being investigated?"
            ]

    # 3. Check for extremely short, underspecified queries
    words = [w for w in re.findall(r"[a-z0-9]+", clean_topic) if len(w) > 2]
    if len(words) < 3 and not any(k in clean_topic for k in ("llm", "rag", "ai", "ml", "dna", "crispr", "quantum")):
        return True, "Research topic is too brief to form a testable hypothesis.", [
            "Could you elaborate on the specific research question or hypothesis?",
            "What specific comparison or outcome are you seeking empirical evidence for?"
        ]

    return False, None, []

