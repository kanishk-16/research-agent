import asyncio
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple

from .evidence_validator import EvidenceValidator
from .extraction.models import (
    ALLOWED_CONFIDENCE,
    ALLOWED_EVIDENCE_TYPES,
    ALLOWED_STANCES,
    ConfidenceType,
    EvidenceExcerptModel,
    EvidenceItemType,
    ExtractedFindingItemModel,
    FindingExtractionModel,
    QuantitativeEvidenceModel,
    StanceType,
)
from .extraction.grounding import (
    STOP_WORDS,
    _clean_and_parse_json,
    _decompose_complex_finding,
    _infer_location_in_source,
    _is_number_grounded,
    _parse_structured_extraction_response,
    _validate_finding_for_question,
    _verify_and_calibrate_stance,
    _verify_passage_relevance,
    _verify_quantitative_grounding,
    extract_heuristic_quantitative_records,
)


def compute_backoff_with_jitter(attempt: int, base_delay: float = 0.2, max_delay: float = 4.0) -> float:
    """Compute exponential backoff delay with full jitter for rate-limit retries."""
    exp_cap = min(max_delay, base_delay * (2 ** attempt))
    return round(random.uniform(base_delay * 0.5, exp_cap), 3)


async def async_exponential_backoff_sleep(attempt: int, base_delay: float = 0.2, max_delay: float = 4.0) -> float:
    """Non-blocking asynchronous exponential backoff sleep with jitter."""
    delay = compute_backoff_with_jitter(attempt, base_delay=base_delay, max_delay=max_delay)
    await asyncio.sleep(delay)
    return delay


MODEL = "gemini-3.8-flash"


def _process_pydantic_finding(
    item: ExtractedFindingItemModel,
    question: dict,
    source_id: str,
    content_to_use: str,
) -> Optional[dict]:
    """
    Convert a Pydantic-validated ExtractedFindingItemModel into an EVL-validated finding dictionary.
    """
    question_id = str(question.get("id", "")).strip().upper()
    question_text = str(question.get("question", "")).strip()
    requires_quant = bool(question.get("requires_quantitative_evidence", False))

    claim = item.claim.strip()
    if not claim:
        return None

    clean_ev = []
    for ev in item.evidence:
        ev_text = ev.evidence_text.strip()
        if not ev_text:
            continue
        loc_str = ev.location_in_source or _infer_location_in_source(ev_text, content_to_use)
        clean_ev.append({
            "source_id": source_id,
            "evidence_type": ev.evidence_type,
            "evidence_text": ev_text,
            "location_in_source": loc_str,
        })

    ev_texts = [e["evidence_text"] for e in clean_ev]
    if not _verify_passage_relevance(claim, ev_texts, question_text):
        return None
    if not _validate_finding_for_question(claim, ev_texts, question):
        return None

    stance = _verify_and_calibrate_stance(item.stance, claim, ev_texts)

    clean_quant = []
    if requires_quant:
        for qitem in item.quantitative_evidence:
            clean_qrecord = qitem.model_dump()
            clean_qrecord["source_id"] = source_id
            if not clean_qrecord.get("location_in_source"):
                clean_qrecord["location_in_source"] = (
                    (clean_ev[0]["location_in_source"] if clean_ev else None)
                    or _infer_location_in_source(str(clean_qrecord.get("metric", "")), content_to_use)
                )
            clean_qrecord = _verify_quantitative_grounding(clean_qrecord, content_to_use)
            clean_quant.append(clean_qrecord)

        if not clean_quant:
            combined_text = f"{claim} " + " ".join(ev_texts)
            clean_quant.extend(
                extract_heuristic_quantitative_records(combined_text, source_id, content_to_use)
            )

    candidate_finding = {
        "question_id": question_id,
        "source_ids": [source_id],
        "claim": claim,
        "stance": stance,
        "confidence": item.confidence,
        "evidence": clean_ev,
        "quantitative_evidence": clean_quant,
    }

    is_valid, validated_f, _ = EvidenceValidator.validate_finding(
        candidate_finding,
        question,
        source_text=content_to_use,
    )
    return validated_f if is_valid else None


def extract_evidence_from_source(
    gemini_client,
    question,
    source,
    model=MODEL
):
    """
    Extract source-grounded structured evidence records for one (question, source) pair
    using Gemini token-level constrained decoding (response_schema=FindingExtractionModel).
    """
    question_id = str(question.get("id", "")).strip().upper()
    question_text = str(question.get("question", "")).strip()
    evidence_needed = question.get("evidence_needed", [])
    requires_quant = bool(question.get("requires_quantitative_evidence", False))
    quant_fields = question.get("quantitative_fields", [])
    exp_fields = question.get("experimental_context_fields", [])
    boundary_conds = question.get("boundary_conditions", [])

    source_id = str(source.get("source_id", "")).strip().upper()
    retrieved = source.get("retrieved_content", "").strip()
    snippet = source.get("content", "").strip()
    content_to_use = retrieved if retrieved else snippet

    if not content_to_use:
        return []

    try:
        from .semantic_indexer import chunk_and_filter_document
        if len(content_to_use) > 3000:
            content_to_use = chunk_and_filter_document(
                content_to_use,
                query=question_text,
                top_k=4,
                min_chars=2500
            )
        content_to_use = content_to_use[:8000]

        prompt = (
            f"Extract source-grounded findings from document {source_id} for research question {question_id}.\n"
            f"Question: {question_text}\n"
            f"Evidence Needed: {evidence_needed} | Quantitative Required: {requires_quant} ({quant_fields})\n"
            f"Context Fields: {exp_fields} | Boundary Conditions: {boundary_conds}\n"
            f"Source Title: {source.get('title', '')} ({source.get('source_type', 'other')})\n\n"
            f"CONTENT:\n{content_to_use}\n\n"
            "Instructions:\n"
            "1. Extract only factual findings explicitly supported by CONTENT with verbatim evidence_text quotes.\n"
            "2. Classify stance directly as 'support', 'counter', 'limitation', 'trade_off', 'mixed', or 'context'.\n"
            "3. Extract distinct findings for supporting evidence, counter-evidence/refutations, and trade-offs.\n"
            "4. Populate quantitative_evidence with grounded numeric values whenever measurements appear."
        )

        response = None
        max_attempts = 1 if type(gemini_client).__name__ == "_DualModeGeminiClient" else 3
        for attempt in range(max_attempts):
            try:
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": FindingExtractionModel,
                    }
                )
                break
            except Exception as err:
                err_str = str(err)
                if attempt + 1 < max_attempts and any(c in err_str for c in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL", "DEADLINE_EXCEEDED", "overloaded", "ServiceUnavailable")):
                    time.sleep(compute_backoff_with_jitter(attempt))
                else:
                    raise err

        extraction_model = _parse_structured_extraction_response(response)
        validated_findings = []
        for item in extraction_model.findings:
            validated_f = _process_pydantic_finding(item, question, source_id, content_to_use)
            if validated_f is not None:
                validated_findings.append(validated_f)

        return validated_findings

    except Exception as error:
        print(f"Evidence extraction failed for {source_id} on {question_id}: {error}")
        return []


def extract_evidence_from_sources_batch(
    gemini_client,
    question,
    sources,
    model=MODEL
) -> List[Tuple[dict, str]]:
    """
    Batch-extract structured evidence for one research question across all its selected sources
    using Gemini token-level constrained decoding (response_schema=FindingExtractionModel).
    """
    from .semantic_indexer import chunk_and_filter_document

    question_id = str(question.get("id", "")).strip().upper()
    question_text = str(question.get("question", "")).strip()
    evidence_needed = question.get("evidence_needed", [])
    requires_quant = bool(question.get("requires_quantitative_evidence", False))
    quant_fields = question.get("quantitative_fields", [])
    exp_fields = question.get("experimental_context_fields", [])
    boundary_conds = question.get("boundary_conditions", [])

    source_texts = {}
    source_blocks = []
    valid_sids = []

    for src in sources:
        sid = str(src.get("source_id", "")).strip().upper()
        retrieved = src.get("retrieved_content", "").strip()
        snippet = src.get("content", "").strip()
        raw_text = retrieved if retrieved else snippet
        if not raw_text or not sid:
            continue
        focused = (
            chunk_and_filter_document(raw_text, query=question_text, top_k=3, min_chars=2000)
            if len(raw_text) > 2500
            else raw_text
        )[:4000]
        source_texts[sid] = raw_text[:15000]
        valid_sids.append(sid)
        source_blocks.append(
            f"=== SOURCE ID: {sid} ===\n"
            f"Title: {src.get('title', '')}\n"
            f"Type: {src.get('source_type', 'other')}\n"
            f"Content:\n{focused}\n"
        )

    if not source_blocks:
        return []

    combined_sources_str = "\n".join(source_blocks)
    allowed_sids_str = ", ".join(f'"{s}"' for s in valid_sids)

    prompt = (
        f"Extract source-grounded findings for research question {question_id} across sources: {allowed_sids_str}.\n"
        f"Question: {question_text}\n"
        f"Evidence Needed: {evidence_needed} | Quantitative Required: {requires_quant} ({quant_fields})\n"
        f"Context Fields: {exp_fields} | Boundary Conditions: {boundary_conds}\n\n"
        f"SOURCES:\n{combined_sources_str}\n"
        "Instructions:\n"
        f"1. Set source_id on each finding to one of: {allowed_sids_str}.\n"
        "2. Classify stance directly as 'support', 'counter', 'limitation', 'trade_off', 'mixed', or 'context'.\n"
        "3. Include verbatim quotes in evidence_text and grounded numbers in quantitative_evidence."
    )

    try:
        response = None
        max_attempts = 1 if type(gemini_client).__name__ == "_DualModeGeminiClient" else 3
        for attempt in range(max_attempts):
            try:
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": FindingExtractionModel,
                    }
                )
                break
            except Exception as err:
                err_str = str(err)
                if attempt + 1 < max_attempts and any(c in err_str for c in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL", "DEADLINE_EXCEEDED", "overloaded", "ServiceUnavailable")):
                    time.sleep(compute_backoff_with_jitter(attempt))
                else:
                    raise err

        extraction_model = _parse_structured_extraction_response(response)
        results_out: List[Tuple[dict, str]] = []
        for item in extraction_model.findings:
            sid = str(item.source_id or "").strip().upper()
            if sid not in source_texts:
                sid = valid_sids[0]
            validated_f = _process_pydantic_finding(item, question, sid, source_texts[sid])
            if validated_f is not None:
                for af in _decompose_complex_finding(validated_f):
                    results_out.append((af, sid))

        return results_out
    except Exception as error:
        print(f"Batch evidence extraction failed on {question_id}: {error}")
        return []


def extract_research_evidence(
    gemini_client,
    research_plan,
    selection_bundle,
    model=MODEL,
    existing_extraction_bundle: Optional[dict] = None
):
    """
    Run evidence extraction across all research questions and selected sources.
    Applies atomic finding decomposition and global multi-question evidence routing.
    Assigns stable finding IDs (Q1-F1, Q1-F2...) and constructs question-level findings and gaps.
    Preserves and merges findings across research rounds when existing_extraction_bundle is provided.
    """

    questions = research_plan.get("questions", [])
    per_question_sel = selection_bundle.get("per_question_selection", {})

    source_finding_map = {}
    all_findings_list = []
    question_raw_findings = {}

    existing_q_findings = {}
    already_extracted_sources_by_qid = {}
    if existing_extraction_bundle:
        for sid, fids in existing_extraction_bundle.get("source_finding_map", {}).items():
            source_finding_map[sid] = list(fids)

        for q_entry in existing_extraction_bundle.get("questions", []):
            qid = str(q_entry.get("question_id", "")).strip().upper()
            findings = list(q_entry.get("findings", []))
            existing_q_findings[qid] = findings
            already_extracted_sources_by_qid[qid] = set()
            for f in findings:
                for sid in f.get("source_ids", []):
                    already_extracted_sources_by_qid[qid].add(sid)

    # Stage 1: Concurrent direct extraction and atomic decomposition across questions
    def _extract_for_one_question(question: dict) -> Tuple[str, List[Tuple[dict, str]]]:
        question_id = str(question.get("id", "")).strip().upper()
        q_sel_data = per_question_sel.get(question_id, {})
        sel_tuples = q_sel_data.get("selected_sources", [])

        sources_to_extract = []
        for source, _ in sel_tuples:
            source_id = source["source_id"]
            source_aliases = set(source.get("all_source_ids", [source_id])) | {source_id}
            if already_extracted_sources_by_qid.get(question_id) and (source_aliases & already_extracted_sources_by_qid[question_id]):
                continue
            sources_to_extract.append(source)

        direct_findings: List[Tuple[dict, str]] = []
        if not sources_to_extract:
            return question_id, direct_findings

        if len(sources_to_extract) > 1 and type(gemini_client).__name__ == "_DualModeGeminiClient":
            direct_findings = extract_evidence_from_sources_batch(
                gemini_client,
                question,
                sources_to_extract,
                model=model
            )
        else:
            for source in sources_to_extract:
                source_id = source["source_id"]
                raw_findings = extract_evidence_from_source(
                    gemini_client,
                    question,
                    source,
                    model=model
                )
                for f in raw_findings:
                    for af in _decompose_complex_finding(f):
                        direct_findings.append((af, source_id))
        return question_id, direct_findings

    if len(questions) > 1:
        with ThreadPoolExecutor(max_workers=min(4, len(questions))) as pool:
            for qid, direct_findings in pool.map(_extract_for_one_question, questions):
                question_raw_findings[qid] = direct_findings
    else:
        for question in questions:
            qid, direct_findings = _extract_for_one_question(question)
            question_raw_findings[qid] = direct_findings

    # Stage 2: Global multi-question evidence routing
    global_findings_pool = []
    for qid, q_findings in question_raw_findings.items():
        for f, sid in q_findings:
            global_findings_pool.append((f, sid, qid))

    # Also include existing findings for cross-question routing opportunities
    for qid, f_list in existing_q_findings.items():
        for f in f_list:
            sids = f.get("source_ids", [""])
            sid = sids[0] if sids else ""
            global_findings_pool.append((f, sid, qid))

    extracted_questions = []

    for question in questions:
        question_id = str(question.get("id", "")).strip().upper()
        q_findings = []
        q_counter_findings = []

        def _is_cross_routing_allowed(orig_q, target_q, finding_item) -> bool:
            t_text = str(target_q.get("question", "")).lower()
            t_type = str(target_q.get("type", "")).lower()
            claim_lower = str(finding_item.get("claim", "")).lower()

            is_defense_target = any(k in t_text for k in ("defense", "mitigat", "guardrail", "sanitiz", "filter", "protect", "robustness", "safeguard")) or "defense" in t_type
            is_mechanics_target = any(k in t_text for k in ("vector", "mechanic", "how do", "threat model", "taxonomy")) or "mechanisms" in t_type

            finding_evaluates_defense = any(k in claim_lower for k in ("defense", "mitigat", "guardrail", "sanitiz", "filter", "pra-rag", "detector", "detection", "perplexity", "refusal", "certified", "resilience", "countermeasure", "false positive"))

            # If target is defense evaluation: finding MUST explicitly evaluate defenses or mitigations
            if is_defense_target and not finding_evaluates_defense:
                return False

            # If target is attack mechanics: pure defense finding without mechanics is disallowed
            if is_mechanics_target and finding_evaluates_defense and not any(k in claim_lower for k in ("vector", "mechanic", "payload", "poison", "injection", "exploit")):
                return False

            # Generalized Topic Gating:
            is_speedup_comp = any(k in t_text for k in ("speedup", "speedup factor", "latency reduction", "across different draft", "draft model size"))
            is_hardware_bottleneck = any(k in t_text for k in ("hardware bottleneck", "architectural bottleneck", "memory bandwidth", "hardware setup", "cpu/gpu", "accelerator"))
            is_threshold_target = any(k in t_text for k in ("threshold", "acceptance rate", "under what", "fail to provide")) or "boundary_conditions" in t_type
            is_tradeoff_failure_target = any(k in t_text for k in ("trade-off", "tradeoff", "failure mode", "side effect", "drawback", "risk")) or "limitations" in t_type

            # If target is hardware bottleneck question: finding must discuss hardware/architecture/bottlenecks
            if is_hardware_bottleneck and not any(k in claim_lower for k in ("bottleneck", "hardware", "cpu", "gpu", "bandwidth", "memory", "cache", "interconnect", "kernel", "communication", "parallelism", "architecture")):
                return False

            # If target is trade-offs / failure modes / limitations question: finding must discuss trade-offs, degradation, overhead, failure, or limitations
            if is_tradeoff_failure_target and not is_hardware_bottleneck and not any(k in claim_lower for k in ("trade-off", "tradeoff", "fail", "degrad", "deteriorat", "overhead", "cost", "drop", "inconsistent", "limitation", "drawback", "penalty", "side effect", "ineffective")):
                return False

            # If target is speedup/latency comparative question: hardware background without speedup or draft model comparison cannot route
            if is_speedup_comp and not any(k in claim_lower for k in ("speedup", "speed-up", "latency reduction", "draft", "x speedup", "accelerat", "throughput")):
                return False

            # If target is threshold / failure question: finding must discuss failure, degradation, threshold, or acceptance rate
            if is_threshold_target and not any(k in claim_lower for k in ("threshold", "acceptance", "fail", "negat", "drop", "slower", "degrad", "overhead", "rejection", "accuracy")):
                return False

            return True

        def _is_semantically_equivalent_claim(c1: str, c2: str, threshold: float = 0.65) -> bool:
            s1 = c1.strip().lower()
            s2 = c2.strip().lower()
            if s1 == s2:
                return True
            for acr, exp in [("asr", "attack success rate"), ("em", "exact match"), ("llm", "large language model"), ("rag", "retrieval augmented generation")]:
                s1 = re.sub(rf"\b{acr}\b", exp, s1)
                s2 = re.sub(rf"\b{acr}\b", exp, s2)
            stop = {
                "the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
                "and", "or", "for", "with", "on", "at", "by", "that", "this", "it"
            }
            tokens1 = {w for w in re.findall(r"\w+", s1) if w not in stop and len(w) > 2}
            tokens2 = {w for w in re.findall(r"\w+", s2) if w not in stop and len(w) > 2}
            if not tokens1 or not tokens2:
                return False
            inter = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            jaccard = inter / union if union > 0 else 0.0
            containment = inter / min(len(tokens1), len(tokens2))
            return max(jaccard, containment) >= threshold

        def _try_merge_or_add_finding(candidate, sid, orig_qid=None, is_pre_existing=False):
            c_claim = candidate.get("claim", "").strip()
            if not c_claim:
                return

            # Check for semantic equivalence with existing findings
            for existing in q_findings:
                if _is_semantically_equivalent_claim(c_claim, existing.get("claim", "")):
                    # Merge source IDs into existing finding
                    if sid and sid not in existing.setdefault("source_ids", []):
                        existing["source_ids"].append(sid)
                    for s in candidate.get("source_ids", []):
                        if s and s not in existing["source_ids"]:
                            existing["source_ids"].append(s)

                    # Merge quotes without duplicates
                    existing_quotes = {e.get("evidence_text", "").strip() for e in existing.get("evidence", [])}
                    for ev in candidate.get("evidence", []):
                        ev_t = ev.get("evidence_text", "").strip()
                        if ev_t and ev_t not in existing_quotes:
                            existing.setdefault("evidence", []).append(ev)
                            existing_quotes.add(ev_t)

                    # Merge quantitative metrics without duplicates
                    existing_metrics = {str(q.get("metric", "")) for q in existing.get("quantitative_evidence", [])}
                    for q in candidate.get("quantitative_evidence", []):
                        m = str(q.get("metric", ""))
                        if m not in existing_metrics:
                            existing.setdefault("quantitative_evidence", []).append(q)
                            existing_metrics.add(m)

                    fid = existing.get("finding_id")
                    if sid:
                        source_finding_map.setdefault(sid, [])
                        if fid and fid not in source_finding_map[sid]:
                            source_finding_map[sid].append(fid)
                    return

            # Enforce Source Concentration Cap for NEW findings only: max 2 findings per unique paper per question
            if not is_pre_existing and sid:
                existing_count_for_sid = sum(1 for existing in q_findings if sid in existing.get("source_ids", []))
                if existing_count_for_sid >= 2:
                    return

            # Assign finding ID: preserve existing finding_id if available and not conflicting
            existing_fids = {f.get("finding_id") for f in q_findings}
            cand_fid = candidate.get("finding_id")
            if is_pre_existing and cand_fid and cand_fid not in existing_fids:
                fid = cand_fid
            else:
                fid = f"{question_id}-F{len(q_findings) + 1}"

            new_f = dict(candidate)
            new_f["finding_id"] = fid
            new_f["question_id"] = question_id
            if orig_qid:
                new_f["routed_from_question"] = orig_qid
            new_f.setdefault("source_ids", [])
            if sid and sid not in new_f["source_ids"]:
                new_f["source_ids"].append(sid)
            for s in candidate.get("source_ids", []):
                if s and s not in new_f["source_ids"]:
                    new_f["source_ids"].append(s)

            for s in new_f.get("source_ids", []):
                source_finding_map.setdefault(s, [])
                if fid not in source_finding_map[s]:
                    source_finding_map[s].append(fid)

            q_findings.append(new_f)
            if new_f not in all_findings_list:
                all_findings_list.append(new_f)
            if new_f.get("stance") == "counter":
                q_counter_findings.append(new_f)

        # 0. Seed existing findings for this question from previous rounds
        for existing_f in existing_q_findings.get(question_id, []):
            first_sid = existing_f.get("source_ids", [None])[0] if existing_f.get("source_ids") else None
            _try_merge_or_add_finding(existing_f, first_sid, is_pre_existing=True)

        # 1. Add direct findings for this question
        for f, sid in question_raw_findings.get(question_id, []):
            _try_merge_or_add_finding(f, sid)

        # 2. Cross-route relevant findings from other questions (Global Evidence Routing)
        for other_f, sid, orig_qid in global_findings_pool:
            if orig_qid == question_id:
                continue

            # Archetype gating: prevent incompatible cross-routing (e.g. pure attack into defense question)
            if not _is_cross_routing_allowed(orig_qid, question, other_f):
                continue

            # Strict EVL gate on candidate finding for this target question
            is_valid, validated_routed, reasons = EvidenceValidator.validate_finding(
                other_f,
                question,
                source_text=""
            )
            if is_valid:
                # If target question requires quantitative evidence and routed finding has no valid quantitative records for it,
                # re-extract heuristic quantitative records tailored to this target question from the finding's text
                if question.get("requires_quantitative_evidence") and not validated_routed.get("quantitative_evidence"):
                    comb_text = f"{validated_routed.get('claim', '')} " + " ".join(e.get("evidence_text", "") for e in validated_routed.get("evidence", []))
                    parsed_quants = extract_heuristic_quantitative_records(comb_text, sid)
                    if parsed_quants:
                        q_ok, clean_parsed, _ = EvidenceValidator.validate_quantitative_validity(
                            {"quantitative_evidence": parsed_quants},
                            question,
                            ""
                        )
                if question.get("requires_quantitative_evidence") and not validated_routed.get("quantitative_evidence") and validated_routed.get("stance") != "counter":
                    # Do not route non-quantitative non-counter findings into a question requiring quantitative evidence
                    continue

                _try_merge_or_add_finding(validated_routed, sid, orig_qid=orig_qid)

        # Determine evidence readiness status
        req_quant = bool(question.get("requires_quantitative_evidence", False))
        req_counter = bool(question.get("requires_counter_evidence", False))

        has_quant = any(len(f.get("quantitative_evidence", [])) > 0 for f in q_findings)
        has_counter = len(q_counter_findings) > 0

        gaps = []
        if not q_findings:
            gaps.append("No findings extracted from selected sources for this question.")
        if req_quant and not has_quant:
            gaps.append("Quantitative evidence required by Planner, but no quantitative records extracted.")
        if req_counter and not has_counter:
            gaps.append("Counter evidence required by Planner, but no actual counter stance findings extracted.")

        if not q_findings:
            status = "insufficient"
        elif (req_quant and not has_quant) or (req_counter and not has_counter):
            status = "partial"
        else:
            status = "complete_candidate"

        extracted_questions.append({
            "question_id": question_id,
            "question": question.get("question", ""),
            "status": status,
            "findings": q_findings,
            "counter_evidence": q_counter_findings,
            "evidence_gaps": gaps
        })

    return {
        "questions": extracted_questions,
        "source_finding_map": source_finding_map,
        "all_findings": all_findings_list
    }
