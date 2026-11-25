"""
search_agent/extraction/models.py

Pydantic structured output schemas for token-level constrained decoding via Gemini SDK
(`response_schema=FindingExtractionModel`).
"""

from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


ALLOWED_STANCES = {"support", "counter", "limitation", "trade_off", "mixed", "context"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
ALLOWED_EVIDENCE_TYPES = {
    "quantitative",
    "qualitative",
    "methodological",
    "boundary_condition",
    "comparison"
}

StanceType = Literal["support", "counter", "limitation", "trade_off", "mixed", "context"]
ConfidenceType = Literal["high", "medium", "low"]
EvidenceItemType = Literal["quantitative", "qualitative", "methodological", "boundary_condition", "comparison"]


class EvidenceExcerptModel(BaseModel):
    """Pydantic model for a single grounded evidence quote/excerpt."""
    evidence_type: EvidenceItemType = Field(
        default="qualitative",
        description="Category of evidence excerpt."
    )
    evidence_text: str = Field(
        default="",
        description="Verbatim supporting quote or tightly grounded excerpt from the source text."
    )
    location_in_source: Optional[str] = Field(
        default=None,
        description="Section, Table, or Figure identifier if discernible (e.g. 'Section 4.1', 'Table 2')."
    )

    @field_validator("evidence_type", mode="before")
    @classmethod
    def _normalize_evidence_type(cls, v: Any) -> str:
        val = str(v or "qualitative").strip().lower()
        return val if val in ALLOWED_EVIDENCE_TYPES else "qualitative"

    @field_validator("location_in_source", mode="before")
    @classmethod
    def _normalize_location(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return None if not s or s.lower() in ("null", "none") else s


class QuantitativeEvidenceModel(BaseModel):
    """Pydantic model for a structured quantitative benchmark or empirical measurement."""
    model: Optional[str] = Field(default=None, description="Evaluated model or system name if specified.")
    dataset: Optional[str] = Field(default=None, description="Dataset or benchmark name if specified.")
    task: Optional[str] = Field(default=None, description="Evaluated task name if specified.")
    metric: Optional[str] = Field(default=None, description="Metric name with unit (e.g. 'Accuracy %', 'Latency (ms)').")
    baseline_score: Optional[float] = Field(default=None, description="Numeric baseline score if explicitly reported.")
    experimental_score: Optional[float] = Field(default=None, description="Numeric experimental or proposed method score.")
    absolute_difference: Optional[float] = Field(default=None, description="Absolute numeric difference if reported or computable.")
    relative_difference: Optional[float] = Field(default=None, description="Relative percentage difference if reported or computable.")
    multiplier: Optional[str] = Field(default=None, description="Multiplier or fold factor if reported (e.g. '3.5x', '10-fold').")
    range_min: Optional[float] = Field(default=None, description="Minimum value of a reported numeric range.")
    range_max: Optional[float] = Field(default=None, description="Maximum value of a reported numeric range.")
    location_in_source: Optional[str] = Field(default=None, description="Table, Figure, or Section where the number appears.")

    @field_validator("model", "dataset", "task", "metric", "multiplier", "location_in_source", mode="before")
    @classmethod
    def _clean_optional_str(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return None if not s or s.lower() in ("null", "none") else s

    @field_validator(
        "baseline_score", "experimental_score", "absolute_difference",
        "relative_difference", "range_min", "range_max",
        mode="before"
    )
    @classmethod
    def _coerce_optional_float(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip().rstrip("%").strip()
            if not s or s.lower() in ("null", "none"):
                return None
            try:
                return float(s)
            except ValueError:
                return None
        return None


class ExtractedFindingItemModel(BaseModel):
    """Pydantic model for a single extracted finding with direct structured LLM stance classification."""
    source_id: Optional[str] = Field(default=None, description="Source ID of the document this finding belongs to.")
    claim: str = Field(default="", description="Concise factual summary statement grounded strictly in the source text.")
    stance: StanceType = Field(
        default="context",
        description="Direct stance classification relative to the research question ('support', 'counter', 'limitation', 'trade_off', 'mixed', 'context')."
    )
    confidence: ConfidenceType = Field(default="medium", description="Epistemic confidence level ('high', 'medium', 'low').")
    evidence: List[EvidenceExcerptModel] = Field(default_factory=list, description="Verbatim supporting excerpts from the source.")
    quantitative_evidence: List[QuantitativeEvidenceModel] = Field(
        default_factory=list,
        description="Structured quantitative metrics grounded in the source text."
    )

    @field_validator("stance", mode="before")
    @classmethod
    def _normalize_stance(cls, v: Any) -> str:
        val = str(v or "context").strip().lower()
        return val if val in ALLOWED_STANCES else "context"

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v: Any) -> str:
        val = str(v or "medium").strip().lower()
        return val if val in ALLOWED_CONFIDENCE else "medium"


class FindingExtractionModel(BaseModel):
    """Top-level Pydantic schema enforced via Gemini SDK response_schema=FindingExtractionModel."""
    findings: List[ExtractedFindingItemModel] = Field(
        default_factory=list,
        description="List of source-grounded findings answering the target research question."
    )
