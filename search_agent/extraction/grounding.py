"""
search_agent/extraction/grounding.py

Provenance inference, quantitative number grounding, atomic finding decomposition,
and structured response parsing.
"""

import json
import re
from typing import List

from ..semantic_indexer import stem_word
from .models import (
    ALLOWED_STANCES,
    FindingExtractionModel,
    QuantitativeEvidenceModel,
)


STOP_WORDS = {
    "what", "which", "where", "when", "whom", "how", "does", "with", "from", "that",
    "this", "each", "were", "been", "have", "more", "over", "into", "their", "such",
    "than", "also", "then", "under", "most", "about", "both", "these", "those", "using",
    "based", "study", "paper", "show", "shows", "shown", "within", "between",
    "among", "across", "will", "would", "could", "should", "some", "other"
}


def _verify_and_calibrate_stance(stance: str, claim: str, evidence_texts: list) -> str:
    """
    Validate the LLM's structured stance classification against domain-agnostic semantic polarity.
    """
    stance_clean = str(stance or "context").strip().lower()
    if stance_clean not in ALLOWED_STANCES:
        stance_clean = "context"

    claim_lower = str(claim or "").lower()
    combined = (f"{claim_lower} " + " ".join(evidence_texts or [])).lower()

    positive_outcomes = (
        "outperform", "outperforms", "outperforming", "superior",
        "state-of-the-art", "sota", "surpasses", "exceeds",
        "reduces hallucination", "reduces error", "mitigates"
    )
    negative_outcomes = (
        "suffered", "error propagation", "lower accuracy", "underperform",
        "underperforms", "degrade", "degrades", "degradation", "fails to", "failed to",
        "worse than", "slower than", "deteriorate", "deteriorates",
        "no significant difference", "increase error", "equivalent compute",
        "normalized compute", "standard prompting matches"
    )
    contrast_conjunctions = (" but ", " however ", "at the cost of", "at the expense of", "trade-off", "tradeoff")
    cost_penalties = ("overhead", "latency", "token cost", "increasing token costs")

    has_pos = any(p in combined for p in positive_outcomes)
    has_neg = any(n in combined for n in negative_outcomes)
    has_contrast = any(c in combined for c in contrast_conjunctions) and any(cp in combined for cp in cost_penalties)

    if has_pos and (has_contrast or has_neg):
        return "mixed"
    if has_neg and not has_pos:
        return "counter"
    if has_pos and not has_neg:
        return "support"

    return stance_clean


def _infer_location_in_source(evidence_text: str, source_text: str) -> str:
    """Infer section, table, or paper region if location_in_source was omitted."""
    if not evidence_text or not source_text:
        return "Main Text"
    search_sub = evidence_text[:40].strip()
    idx = source_text.lower().find(search_sub.lower()) if search_sub else -1
    if idx != -1:
        preceding = source_text[max(0, idx - 500):idx]
        sec_match = re.findall(r"(?:Section|Sec\.|Table|Tab\.|Figure|Fig\.|Appendix)\s+[0-9A-Za-z\.]+", preceding, re.I)
        if sec_match:
            return sec_match[-1].strip()
        if "abstract" in preceding.lower():
            return "Abstract"
        if "introduction" in preceding.lower():
            return "Introduction"
        if "conclusion" in preceding.lower():
            return "Conclusion"
    return "Main Text"


def _decompose_complex_finding(finding: dict) -> list[dict]:
    """
    Decomposes a compound finding into atomic findings when it bundles both
    positive performance gains and negative trade-offs / bottlenecks / overhead.
    """
    claim = str(finding.get("claim", "")).strip()
    quant_list = finding.get("quantitative_evidence", [])

    contrast_match = re.search(
        r"^(.*?)\s*(?:,\s*(?:but|however|while|yet|whereas)\s*|\s+(?:at the cost of|accompanied by|offset by)\s+)(.*)$",
        claim,
        re.I
    )

    if not contrast_match and len(quant_list) <= 1:
        return [finding]

    if contrast_match:
        part1 = contrast_match.group(1).strip()
        part2 = contrast_match.group(2).strip()

        if len(part1) > 15 and len(part2) > 15:
            quant1 = []
            quant2 = []
            for qrec in quant_list:
                metric_name = str(qrec.get("metric", "")).lower()
                if any(m in metric_name for m in ("cost", "latency", "token", "overhead", "time", "error")):
                    quant2.append(qrec)
                else:
                    quant1.append(qrec)

            stance1 = _verify_and_calibrate_stance("support", part1, [])
            stance2 = _verify_and_calibrate_stance("counter", part2, [])

            f1 = dict(finding)
            f1["claim"] = part1
            f1["stance"] = stance1
            f1["quantitative_evidence"] = quant1 if quant1 else quant_list
            f1["atomic_type"] = "performance"

            f2 = dict(finding)
            f2["claim"] = part2.capitalize()
            f2["stance"] = stance2
            f2["quantitative_evidence"] = quant2 if quant2 else []
            f2["atomic_type"] = "overhead_or_limitation"

            return [f1, f2]

    return [finding]


def _validate_finding_for_question(claim: str, evidence_texts: list, question: dict) -> bool:
    """
    Validate that an extracted finding specifically answers its assigned research question.
    """
    if not claim:
        return False
    claim_lower = claim.lower()
    qtext = str(question.get("question", "")).lower()
    qtype = str(question.get("type", "")).lower()
    combined = f"{claim_lower} " + " ".join(evidence_texts).lower()

    if any(k in qtext for k in ("overhead", "latency", "cost", "token", "compute", "resource", "api cost")):
        cost_indicators = (
            "latency", "cost", "overhead", "token", "compute", "time", "gpu",
            "flops", "api", "inference", "consumption", "throughput", "efficiency",
            "resource", "delay", "expensive", "cheaper", "multiplier", "budget"
        )
        if not any(ind in combined for ind in cost_indicators):
            return False

    if "failure" in qtext or "bottleneck" in qtext or "error propagation" in qtext or "limitations" in qtype:
        limitation_indicators = (
            "fail", "failure", "bottleneck", "error", "propagation", "overhead",
            "latency", "cost", "token", "degrad", "limitation", "breakdown",
            "hallucination", "vulnerability", "groupthink", "loop", "collapse",
            "decay", "cascading", "inefficiency", "conflict"
        )
        if not any(ind in combined for ind in limitation_indicators):
            return False

    if "boundary condition" in qtext or "under what" in qtext or "when" in qtext or "topology" in qtext or "modular" in qtext:
        boundary_indicators = (
            "boundary", "condition", "complexity", "trade-off", "tradeoff",
            "task-dependent", "when", "structure", "topology", "modular",
            "decomposition", "role", "coordination", "heterogeneous",
            "threshold", "scale", "dense", "hierarchy", "centralized"
        )
        if not any(ind in combined for ind in boundary_indicators):
            return False

    if "compare" in qtext or "comparison" in qtype or "quantitatively" in qtext:
        comparative_indicators = (
            "compare", "comparison", "vs", "versus", "outperform", "baseline",
            "single-agent", "multi-agent", "accuracy", "score", "benchmark",
            "higher", "lower", "better", "worse", "gain", "difference",
            "reduction", "mitigate", "improvement", "superior"
        )
        if not any(ind in combined for ind in comparative_indicators):
            return False

    if "security" in qtext or "vulnerability" in qtext or "privacy" in qtext:
        security_indicators = (
            "security", "leakage", "vulnerability", "privacy", "attack",
            "injection", "safety", "trust", "exposure", "jailbreak"
        )
        if not any(ind in combined for ind in security_indicators):
            return False

    return True


def _verify_passage_relevance(claim: str, evidence_texts: list, question_text: str) -> bool:
    """
    Validate that a claim or passage shares substantive vocabulary with the target question.
    """
    if not claim:
        return False

    q_tokens = {w for w in re.findall(r"[a-z0-9]+", question_text.lower()) if len(w) > 2 and w not in STOP_WORDS}
    text_tokens = {w for w in re.findall(r"[a-z0-9]+", (f"{claim} " + " ".join(evidence_texts)).lower()) if len(w) > 2 and w not in STOP_WORDS}

    if q_tokens & text_tokens:
        return True
    q_stems = {stem_word(w) for w in q_tokens}
    t_stems = {stem_word(w) for w in text_tokens}
    return bool(q_stems & t_stems)


def _is_number_grounded(num_val, source_text: str) -> bool:
    if num_val is None:
        return True
    try:
        val_float = float(num_val)
    except (ValueError, TypeError):
        return True
    str_val = str(num_val).strip()
    if str_val in source_text:
        return True
    if str_val.endswith(".0") and str_val[:-2] in source_text:
        return True
    round_str = f"{val_float:.1f}"
    if round_str in source_text:
        return True
    return False


def _verify_quantitative_grounding(qrecord: dict, source_text: str) -> dict:
    """Verify quantitative metrics against source text and auto-compute deltas."""
    b_score = qrecord.get("baseline_score")
    e_score = qrecord.get("experimental_score")
    diff = qrecord.get("absolute_difference")

    if b_score == 0.0 and (e_score is not None or diff is not None):
        has_explicit_zero = bool(re.search(r"\b0(?:\.0)?%|\bzero(?:\s+percent)?\b", source_text.lower()))
        if not has_explicit_zero:
            if diff is None and e_score is not None:
                qrecord["absolute_difference"] = e_score
                qrecord["experimental_score"] = None
            qrecord["baseline_score"] = None
            b_score = None
            e_score = qrecord.get("experimental_score")

    if b_score is not None and not _is_number_grounded(b_score, source_text):
        qrecord["baseline_score"] = None
        b_score = None

    if e_score is not None and not _is_number_grounded(e_score, source_text):
        qrecord["experimental_score"] = None
        e_score = None

    if isinstance(b_score, (int, float)) and isinstance(e_score, (int, float)):
        qrecord["absolute_difference"] = round(float(e_score) - float(b_score), 4)
        if float(b_score) != 0:
            qrecord["relative_difference"] = round(
                ((float(e_score) - float(b_score)) / float(b_score)) * 100.0, 2
            )
        else:
            qrecord["relative_difference"] = None
    elif qrecord.get("absolute_difference") is not None:
        if not _is_number_grounded(qrecord["absolute_difference"], source_text):
            qrecord["absolute_difference"] = None
        qrecord["relative_difference"] = None

    multiplier = qrecord.get("multiplier")
    if multiplier is not None:
        m_str = str(multiplier).strip()
        num_m = re.search(r"(\d+(?:\.\d+)?)", m_str)
        if num_m:
            if not _is_number_grounded(num_m.group(1), source_text):
                qrecord["multiplier"] = None
        else:
            qrecord["multiplier"] = None

    rmin = qrecord.get("range_min")
    rmax = qrecord.get("range_max")
    if rmin is not None and not _is_number_grounded(rmin, source_text):
        qrecord["range_min"] = None
    if rmax is not None and not _is_number_grounded(rmax, source_text):
        qrecord["range_max"] = None

    return qrecord


def _clean_and_parse_json(raw_output: str) -> dict:
    """Clean and parse JSON response from LLM."""
    text = raw_output.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()

    first_brace = text.find("{")
    if first_brace != -1:
        text = text[first_brace:]

    decoder = json.JSONDecoder(strict=False)
    for candidate in (
        text,
        re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text),
        re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)),
    ):
        try:
            obj, _ = decoder.raw_decode(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    last_brace = text.rfind("}")
    if last_brace != -1:
        try:
            return json.loads(text[:last_brace + 1], strict=False)
        except Exception:
            pass
    return {}


def extract_heuristic_quantitative_records(text: str, source_id: str, content_to_use: str = "") -> List[dict]:
    """
    Fallback quantitative record builder constructing Pydantic-validated QuantitativeEvidenceModel
    records from text when structured LLM output omitted numeric fields.
    """
    records: List[dict] = []
    text_lower = text.lower()

    def _add_record(**kwargs):
        q = QuantitativeEvidenceModel(location_in_source="Parsed from finding text", **kwargs).model_dump()
        q["source_id"] = source_id
        records.append(q)

    if any(k in text_lower for k in ("hallucinat", "factual error", "faithfulness")):
        pct_metric = "Hallucination / Error Rate %"
    elif any(k in text_lower for k in ("accuracy", "exact match", "f1", "qa score")):
        pct_metric = "Accuracy / Performance Score %"
    elif any(k in text_lower for k in ("asr", "attack success")):
        pct_metric = "Attack Success Rate %"
    else:
        pct_metric = "Empirical Score / Proportion %"

    for vm in re.finditer(r"(?:from\s+)?(\d+(?:\.\d+)?)\s*%\s*(?:vs\.?|compared to|against|to|-)\s*(\d+(?:\.\d+)?)\s*%", text, re.I):
        b_val, e_val = float(vm.group(1)), float(vm.group(2))
        diff = round(e_val - b_val, 4)
        _add_record(metric=pct_metric, baseline_score=b_val, experimental_score=e_val, absolute_difference=diff, relative_difference=round((diff / b_val) * 100.0, 2) if b_val else None)

    for tm in re.finditer(r"(?:from\s+)?(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(microseconds?|μs|us|ms|milliseconds?|seconds?|s|KiB|KB|MB|MiB)\b", text, re.I):
        b_val, e_val, unit = float(tm.group(1)), float(tm.group(2)), tm.group(3)
        diff = round(e_val - b_val, 4)
        m_type = "Serialized Size" if unit.lower() in ("kib", "kb", "mb", "mib") else "Latency / Execution Time"
        _add_record(metric=f"{m_type} ({unit})", baseline_score=b_val, experimental_score=e_val, absolute_difference=diff, relative_difference=round((diff / b_val) * 100.0, 2) if b_val else None)

    for lm in re.finditer(r"(?:latency\s+overhead\s+(?:of\s+)?|overhead\s+(?:of\s+)?|latency\s+(?:of\s+)?|delay\s+(?:of\s+)?)(\d+(?:\.\d+)?)\s*(ms|milliseconds?|microseconds?|μs|us|seconds?|s)\b", text, re.I):
        l_val = float(lm.group(1))
        _add_record(metric=f"Latency Overhead ({lm.group(2)})", experimental_score=l_val, absolute_difference=l_val)

    mm = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|[- ]fold)\s*(?:overhead|speedup|token|cost|latency|increase|gain)?", text, re.I)
    if mm:
        mult_str = f"{mm.group(1)}x" if "fold" not in mm.group(0).lower() else f"{mm.group(1)}-fold"
        _add_record(metric="Multiplier / Overhead Factor", multiplier=mult_str)

    for dm in re.finditer(r"(?:improved?|increased?|decreased?|dropped?|reduced?|by)\s+([\+\-]?\d+(?:\.\d+)?)\s*%", text, re.I):
        _add_record(metric="Reported Difference %", absolute_difference=float(dm.group(1)))

    for pm in re.finditer(r"(?:over|more than|approximately|~|exceeding)?\s*(\d+(?:\.\d+)?)\s*%\s*of\s+([a-zA-Z0-9\s_-]+?)(?:\s+(?:were|are|was|spent|failed|rejected|expensive|overhead)|[,\.]|$)", text, re.I):
        p_val = float(pm.group(1))
        cat = pm.group(2).strip()[:40].strip()
        _add_record(metric=f"Proportion of {cat.title()} %", experimental_score=p_val, absolute_difference=p_val)

    for am in re.finditer(r"(?:acceptance\s+rate|rejection\s+rate|threshold)\s*(?:of|below|<|<=|>|>=)?\s*(\d+(?:\.\d+)?)\s*(%|percent)?\b", text, re.I):
        rate_val = float(am.group(1))
        unit = "%" if am.group(2) or rate_val > 1.0 else "ratio"
        _add_record(metric=f"Acceptance / Failure Threshold ({unit})", experimental_score=rate_val, absolute_difference=rate_val)

    if content_to_use:
        records = [_verify_quantitative_grounding(r, content_to_use) for r in records]
    return records


def _parse_structured_extraction_response(response) -> FindingExtractionModel:
    """
    Parse Gemini response into a validated FindingExtractionModel instance.
    """
    if response is None:
        return FindingExtractionModel(findings=[])
    parsed_obj = getattr(response, "parsed", None)
    if isinstance(parsed_obj, FindingExtractionModel):
        return parsed_obj
    if isinstance(parsed_obj, dict):
        try:
            return FindingExtractionModel.model_validate(parsed_obj)
        except Exception:
            pass
    raw_text = getattr(response, "text", None)
    if not raw_text or not isinstance(raw_text, str):
        return FindingExtractionModel(findings=[])
    try:
        return FindingExtractionModel.model_validate_json(raw_text.strip())
    except Exception:
        raw_dict = _clean_and_parse_json(raw_text)
        if isinstance(raw_dict, dict):
            try:
                return FindingExtractionModel.model_validate(raw_dict)
            except Exception:
                pass
        return FindingExtractionModel(findings=[])
