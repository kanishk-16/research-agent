import json
import re

MODEL = "gemini-3.5-flash-lite"

ALLOWED_STANCES = {"support", "counter", "mixed", "context"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
ALLOWED_EVIDENCE_TYPES = {
    "quantitative",
    "qualitative",
    "methodological",
    "boundary_condition",
    "comparison"
}


def _clean_and_parse_json(raw_output: str) -> dict:
    """
    Robustly clean and parse JSON response from LLM, handling markdown code fences,
    extraneous text, extra trailing data after JSON object, control characters, and unescaped quotes/newlines.
    """
    text = raw_output.strip()

    # Strip markdown code blocks
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Find the start of the JSON object
    first_brace = text.find("{")
    if first_brace != -1:
        text = text[first_brace:]

    decoder = json.JSONDecoder(strict=False)

    # 1. Try raw_decode directly (parses top-level JSON object and ignores trailing extra data)
    try:
        obj, _ = decoder.raw_decode(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2. Try raw_decode on text cleaned of invalid control characters
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    try:
        obj, _ = decoder.raw_decode(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 3. Try raw_decode after sanitizing unescaped backslashes
    sanitized = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', cleaned)
    try:
        obj, _ = decoder.raw_decode(sanitized)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 4. Fallback: isolate between first '{' and last '}'
    last_brace = text.rfind("}")
    if last_brace != -1:
        try:
            return json.loads(text[:last_brace + 1], strict=False)
        except Exception:
            pass

    return {}


def extract_evidence_from_source(
    gemini_client,
    question,
    source,
    model=MODEL
):
    """
    Extract source-grounded structured evidence records for one (question, source) pair.
    Uses retrieved_content (falling back to content snippet if deep content unavailable).
    """

    question_id = str(question.get("id", "")).strip().upper()
    question_text = str(question.get("question", "")).strip()
    evidence_needed = question.get("evidence_needed", [])
    requires_quant = bool(question.get("requires_quantitative_evidence", False))
    quant_fields = question.get("quantitative_fields", [])
    exp_fields = question.get("experimental_context_fields", [])
    boundary_conds = question.get("boundary_conditions", [])

    source_id = str(source.get("source_id", "")).strip().upper()

    # Content selection: use retrieved_content first, fallback to search snippet content
    retrieved = source.get("retrieved_content", "").strip()
    snippet = source.get("content", "").strip()
    content_to_use = retrieved if retrieved else snippet

    if not content_to_use:
        return []

    try:
        # Bounded text snippet limit to prevent prompt overflows while giving maximum context
        max_chars = 30000
        if len(content_to_use) > max_chars:
            content_to_use = content_to_use[:max_chars]

        prompt = f"""
You are a precise evidence extraction agent in an autonomous research system.
Extract factual findings grounded ONLY in the provided source text for the target research question.

TARGET RESEARCH QUESTION:
ID: {question_id}
Question: {question_text}
Evidence Needed: {evidence_needed}
Requires Quantitative Evidence: {requires_quant}
Quantitative Fields: {quant_fields}
Experimental Context Fields: {exp_fields}
Boundary Conditions to Look For: {boundary_conds}

SOURCE DOCUMENT:
Source ID: {source_id}
Title: {source.get("title", "")}
URL: {source.get("url", "")}
Source Type: {source.get("source_type", "other")}

CONTENT:
{content_to_use}

CRITICAL EXTRACTION RULES:
1. Extract ONLY facts explicitly supported by the text.
2. Do NOT invent claims, numbers, statistics, benchmarks, URLs, or source IDs.
3. Use ONLY source_id "{source_id}". Do NOT generate or reference any other source ID.
4. Stance must be EXACTLY one of: "support", "counter", "mixed", "context".
   - "support": Evidence strengthens or confirms the proposition in the research question.
   - "counter": Evidence contradicts, weakens, or highlights failure modes/limitations.
   - "mixed": Source contains contradictory evidence.
   - "context": Relevant background or methodology that does not directly take a side.
5. Confidence must be EXACTLY one of: "high", "medium", "low".
6. Evidence type must be EXACTLY one of: "quantitative", "qualitative", "methodological", "boundary_condition", "comparison".
7. If no relevant evidence exists in the text for this research question, return an empty array [].

Return ONLY valid JSON matching this exact structure:
{{
  "findings": [
    {{
      "claim": "Concise factual summary statement grounded in the source text.",
      "stance": "support|counter|mixed|context",
      "confidence": "high|medium|low",
      "evidence": [
        {{
          "evidence_type": "quantitative|qualitative|methodological|boundary_condition|comparison",
          "evidence_text": "Short supporting quote or tightly grounded excerpt from text.",
          "location_in_source": "Section, Table, or Paragraph description if discernible, else null"
        }}
      ],
      "quantitative_evidence": [
        {{
          "model": "Model name if mentioned, else null",
          "dataset": "Dataset name if mentioned, else null",
          "task": "Task name if mentioned, else null",
          "metric": "Metric name (e.g. Accuracy, Latency, Hallucination Rate) if mentioned, else null",
          "baseline_score": null,
          "experimental_score": null,
          "absolute_difference": null,
          "relative_difference": null,
          "location_in_source": "Table or section if mentioned, else null"
        }}
      ]
    }}
  ]
}}
"""

        import time

        response = None
        for attempt in range(4):
            try:
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"response_mime_type": "application/json"}
                )
                break
            except Exception as err:
                err_str = str(err)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    time.sleep(12)
                else:
                    raise err

        if response is None or not response.text:
            return []

        raw_output = response.text.strip()
        parsed = _clean_and_parse_json(raw_output)
        raw_findings = parsed.get("findings", [])
        if not isinstance(raw_findings, list):
            return []

        validated_findings = []

        for item in raw_findings:
            if not isinstance(item, dict):
                continue

            claim = str(item.get("claim", "")).strip()
            if not claim:
                continue

            stance = str(item.get("stance", "context")).strip().lower()
            if stance not in ALLOWED_STANCES:
                stance = "context"

            confidence = str(item.get("confidence", "medium")).strip().lower()
            if confidence not in ALLOWED_CONFIDENCE:
                confidence = "medium"

            # Clean evidence items
            raw_ev = item.get("evidence", [])
            clean_ev = []
            if isinstance(raw_ev, list):
                for ev in raw_ev:
                    if isinstance(ev, dict):
                        ev_text = str(ev.get("evidence_text", "")).strip()
                        if not ev_text:
                            continue
                        ev_type = str(ev.get("evidence_type", "qualitative")).strip().lower()
                        if ev_type not in ALLOWED_EVIDENCE_TYPES:
                            ev_type = "qualitative"
                        loc = ev.get("location_in_source")
                        loc_str = str(loc).strip() if loc is not None else None

                        clean_ev.append({
                            "source_id": source_id,
                            "evidence_type": ev_type,
                            "evidence_text": ev_text,
                            "location_in_source": loc_str
                        })

            # Clean quantitative evidence items
            raw_quant = item.get("quantitative_evidence", [])
            clean_quant = []
            if requires_quant and isinstance(raw_quant, list):
                for qitem in raw_quant:
                    if isinstance(qitem, dict):
                        clean_qrecord = {"source_id": source_id}
                        for field in [
                            "model", "dataset", "task", "metric",
                            "baseline_score", "experimental_score",
                            "absolute_difference", "relative_difference",
                            "location_in_source"
                        ]:
                            val = qitem.get(field)
                            if isinstance(val, (int, float)):
                                clean_qrecord[field] = val
                            elif isinstance(val, str) and val.strip():
                                clean_qrecord[field] = val.strip()
                            else:
                                clean_qrecord[field] = None

                        clean_quant.append(clean_qrecord)

            validated_findings.append({
                "question_id": question_id,
                "source_ids": [source_id],
                "claim": claim,
                "stance": stance,
                "confidence": confidence,
                "evidence": clean_ev,
                "quantitative_evidence": clean_quant
            })

        return validated_findings

    except Exception as error:
        print(f"Evidence extraction failed for {source_id} on {question_id}: {error}")
        return []


def extract_research_evidence(
    gemini_client,
    research_plan,
    selection_bundle,
    model=MODEL
):
    """
    Run evidence extraction across all research questions and selected sources.
    Assigns stable finding IDs (Q1-F1, Q1-F2...) and constructs question-level findings and gaps.
    """

    questions = research_plan.get("questions", [])
    per_question_sel = selection_bundle.get("per_question_selection", {})

    extracted_questions = []
    source_finding_map = {}
    all_findings_list = []

    for question in questions:
        question_id = str(question.get("id", "")).strip().upper()
        q_sel_data = per_question_sel.get(question_id, {})
        sel_tuples = q_sel_data.get("selected_sources", [])

        q_findings = []
        q_counter_findings = []
        finding_counter = 1

        for source, reason in sel_tuples:
            source_id = source["source_id"]

            raw_findings = extract_evidence_from_source(
                gemini_client,
                question,
                source,
                model=model
            )

            for f in raw_findings:
                fid = f"{question_id}-F{finding_counter}"
                finding_counter += 1

                f["finding_id"] = fid

                # Map source -> findings
                if source_id not in source_finding_map:
                    source_finding_map[source_id] = []
                if fid not in source_finding_map[source_id]:
                    source_finding_map[source_id].append(fid)

                q_findings.append(f)
                all_findings_list.append(f)

                if f["stance"] == "counter":
                    q_counter_findings.append(f)

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
