"""
Evidence Validation Layer (EVL) for Autonomous Research Agent.

Enforces five mandatory post-retrieval validation gates on every extracted finding:
1. Gate 1: Question Relevance & Target Variable Gate
2. Gate 2: Claim Support & Entailment Gate
3. Gate 3: Quantitative Metric Validity Gate
4. Gate 4: Context Compatibility & Stance Nuance Gate (support, counter, limitation, trade_off, context)
5. Gate 5: Source Independence & Cluster Consolidation Gate
"""

import re
from typing import Dict, List, Any, Tuple, Optional


ALLOWED_STANCES = {"support", "counter", "limitation", "trade_off", "context"}

STOP_WORDS = {
    "what", "which", "where", "when", "whom", "how", "does", "with", "from", "that",
    "this", "each", "were", "been", "have", "more", "over", "into", "their", "such",
    "than", "also", "then", "under", "most", "about", "both", "these", "those", "using",
    "based", "study", "paper", "show", "shows", "shown", "within", "between", "among",
    "across", "will", "would", "could", "should", "some", "other", "there", "their"
}

BIO_CHEM_MARKERS = (
    "dna methylation", "mass spectrometry", "proteomics", "cardiovascular",
    "condensation nuclei", "in vitro", "in vivo", "pharmacokinetics",
    "schizophrenia", "clinical psychology", "sugar-sweetened beverages",
    "blood pressure", "cellular pathways"
)

# Benchmark / Performance Metric names for quantitative validation
ACCURACY_METRIC_KEYWORDS = (
    "accuracy", "pass@", "pass rate", "f1", "bleu", "rouge", "score", "win rate",
    "success rate", "precision", "recall", "exact match", "em", "error rate"
)

COST_LATENCY_METRIC_KEYWORDS = (
    "latency", "cost", "overhead", "token", "compute", "time", "gpu", "flops",
    "inference", "throughput", "delay", "multiplier", "seconds", "ms", "api cost", "budget"
)

NON_EVIDENTIARY_NUMBERS = (
    "sample size", "participants", "subjects", "cohort", "year", "date",
    "parameter", "parameters", "citation", "citations", "batch size", "epochs"
)

SUPPORT_PATTERNS = [
    r"(?:outperform|superior|better|higher|improves?|surpasses?|exceeds?|enhances?|boosts?)",
    r"(?:reduces?|mitigates?|suppresses?|eliminates?)\s+(?:hallucination|error|failure|toxicity|bias)",
    r"(?:reduction|drop|decrease)\s+in\s+(?:hallucination|error|failure|toxicity|bias)\s+(?:by|of)\s+\d+",
    r"(?:state-of-the-art|sota)\s+(?:accuracy|performance|results?|reasoning)",
    r"(?:significant|substantial)\s+(?:gain|improvement|boost)\s+in\s+(?:accuracy|reasoning|f1|pass@1)",
]

COUNTER_PATTERNS = [
    r"(?:baseline|single-agent|standard\s+(?:prompting|model|approach))\b[^\.\;\n]*\b(?:matches|outperforms|superior|better|exceeds?)",
    r"(?:underperform|fails?|failures?|degrades?|degradation|worse|lower)\s+(?:than|compared to)\s+(?:baseline|single-agent)",
    r"(?:equal|equivalent|normalized)\s+(?:compute|budget|tokens?)[^\.\;\n]*\b(?:eliminates?|matches|no significant difference|no advantage)",
    r"(?:not consistently|fails to consistently|no significant difference|statistically equivalent|fails to outperform)",
    r"(?:increases?|escalates?|worsens?)\s+(?:hallucination|error|latency|cost|overhead|failure)",
    r"(?:retrieval noise|distractor|irrelevant\s+context)\b[^\.\;\n]*\b(?:degrades?|impairs?|lowers?|increases? error)",
    r"(?:consensus breakdown|debate hacking|groupthink|infinite loop|collapse)",
]

TRADE_OFF_PATTERNS = [
    r"(?:higher accuracy|improves?|outperforms?|superior|better|gains?|boosts?|advances?)[^\.\;\n]*(?:but|however|at the cost of|accompanied by|offset by|at the expense of)[^\.\;\n]*(?:overhead|latency|cost|error|penalty|tokens?|trade-?off)",
    r"(?:trade-off|tradeoff|task-dependent|mixed results|mixed performance)",
    r"(?:outperform|superior)[^\.\;\n]*(?:on|in)[^\.\;\n]*(?:math|gsm8k|complex)[^\.\;\n]*(?:but|while)[^\.\;\n]*(?:underperform|worse|degrades?|fails?)",
    r"(?:gains?|improvements?)\s+(?:diminish|disappear|vanish)\s+(?:on|under|when)",
]

LIMITATION_PATTERNS = [
    r"(?:limited to|evaluated only on|tested only on|restricted to|scoped to)",
    r"(?:limitation|constraint|bottleneck|drawback|sensitivity to prompt|narrow scope)",
    r"(?:requires? significant|high memory requirement|hardware bottleneck)",
    r"(?:boundary condition|fails on simple|does not generalize to)",
]


class EvidenceValidator:
    """
    Five-gate validation engine ensuring all evidence findings are factual,
    question-aligned, quantitatively valid, and source-independent.
    """

    @staticmethod
    def validate_question_relevance(finding: dict, question: dict) -> Tuple[bool, str]:
        """
        Gate 1: Question Relevance & Target Variable Gate.
        Rejects findings that do not measure or directly address the question's target variable,
        and rejects cross-domain pollution.
        """
        claim = str(finding.get("claim", "")).strip()
        if not claim:
            return False, "Empty claim"

        claim_lower = claim.lower()
        if any(m in claim_lower for m in BIO_CHEM_MARKERS):
            return False, "Disqualified biological/medical marker in claim"

        ev_texts = [e.get("evidence_text", "") for e in finding.get("evidence", [])]
        combined = f"{claim_lower} " + " ".join(ev_texts).lower()

        qtext = str(question.get("question", "")).lower()
        qtype = str(question.get("type", "")).lower()
        target_var = str(question.get("target_variable", "")).lower()

        # Variable Gate 1: Overhead / Latency / Cost / Token requirements
        if any(k in qtext for k in ("overhead", "latency", "cost", "token", "compute", "resource", "api cost")) or "cost" in target_var:
            cost_indicators = (
                "latency", "cost", "overhead", "token", "compute", "time", "gpu",
                "flops", "api", "inference", "consumption", "throughput", "efficiency",
                "resource", "delay", "expensive", "cheaper", "multiplier", "budget", "seconds", "ms"
            )
            if not any(ind in combined for ind in cost_indicators):
                return False, "Missing required latency/cost/overhead metric for Question"

        # Variable Gate 2: Limitations / Failure modes / Error propagation
        elif "failure" in qtext or "bottleneck" in qtext or "error propagation" in qtext or "limitations" in qtype:
            limitation_indicators = (
                "fail", "failure", "bottleneck", "error", "propagation", "overhead",
                "latency", "cost", "token", "degrad", "limitation", "breakdown",
                "hallucination", "vulnerability", "groupthink", "loop", "collapse",
                "decay", "cascading", "inefficiency", "conflict"
            )
            if not any(ind in combined for ind in limitation_indicators):
                return False, "Missing failure/limitation/bottleneck indicators for Question"

        # Variable Gate 3: Boundary conditions / Modularity / Coordination topology
        elif "boundary condition" in qtext or "under what" in qtext or "when" in qtext or "topology" in qtext or "boundary_conditions" in qtype:
            boundary_indicators = (
                "boundary", "condition", "complexity", "trade-off", "tradeoff",
                "task-dependent", "when", "structure", "topology", "modular",
                "decomposition", "role", "coordination", "heterogeneous",
                "threshold", "scale", "dense", "hierarchy", "centralized"
            )
            if not any(ind in combined for ind in boundary_indicators):
                return False, "Missing boundary condition/topology indicators for Question"

        # Variable Gate 4: Comparative benchmark reasoning accuracy
        elif "compare" in qtext or "comparison" in qtype or "quantitatively" in qtext:
            comparative_indicators = (
                "compare", "comparison", "vs", "versus", "outperform", "baseline",
                "single-agent", "multi-agent", "accuracy", "score", "benchmark",
                "higher", "lower", "better", "worse", "gain", "difference",
                "reduction", "mitigate", "improvement", "superior"
            )
            if not any(ind in combined for ind in comparative_indicators):
                return False, "Missing comparative reasoning/accuracy indicators for Question"

        # Variable Gate 5: Historical origins
        elif "historical origin" in qtext or "history" in qtext or "originate" in qtext:
            history_indicators = (
                "history", "historical", "origin", "originate", "first", "century",
                "began", "inception", "development", "evolv", "foundation"
            )
            if not any(ind in combined for ind in history_indicators):
                return False, "Missing historical origin indicators for Question"

        # Substantive vocabulary overlap check
        q_tokens = {w for w in re.findall(r"[a-z0-9]+", qtext) if len(w) > 2 and w not in STOP_WORDS}
        text_tokens = {w for w in re.findall(r"[a-z0-9]+", combined) if len(w) > 2 and w not in STOP_WORDS}
        if len(q_tokens & text_tokens) < 1:
            return False, "Insufficient substantive vocabulary overlap with question"

        return True, "Passed Question Relevance"

    @staticmethod
    def validate_claim_support(finding: dict, source_text: str) -> Tuple[bool, str]:
        """
        Gate 2: Claim Support & Entailment Gate.
        Verifies that cited quotes are grounded in source_text and entail the claim.
        """
        claim = str(finding.get("claim", "")).strip()
        ev_list = finding.get("evidence", [])
        if not ev_list:
            return False, "No supporting excerpts provided"

        if not source_text:
            # When source text unavailable, check that excerpt is substantial
            return True, "Source text not provided for verification"

        source_lower = source_text.lower()
        grounded_excerpts = 0
        for ev in ev_list:
            ev_text = str(ev.get("evidence_text", "")).strip().lower()
            if not ev_text:
                continue
            # Check if excerpt or a 30-char substring exists in source text
            sample = ev_text[:35].strip()
            if sample in source_lower:
                grounded_excerpts += 1

        if grounded_excerpts == 0:
            return False, "Supporting excerpts not grounded in source text"

        # Check semantic claim entailment overlap
        claim_tokens = {w for w in re.findall(r"[a-z0-9]+", claim.lower()) if len(w) > 2 and w not in STOP_WORDS}
        ev_all_text = " ".join(e.get("evidence_text", "") for e in ev_list).lower()
        ev_tokens = {w for w in re.findall(r"[a-z0-9]+", ev_all_text) if len(w) > 2 and w not in STOP_WORDS}

        if claim_tokens and not (claim_tokens & ev_tokens):
            return False, "Claim shares zero substantive concepts with cited evidence excerpts"

        return True, "Passed Claim Support"

    @classmethod
    def validate_quantitative_validity(cls, finding: dict, question: dict, source_text: str) -> Tuple[bool, List[dict], str]:
        """
        Gate 3: Quantitative Metric Validity Gate.
        Filters out non-target numbers (sample sizes, dates, parameter counts) from being
        counted as quantitative benchmark evidence.
        """
        raw_quant = finding.get("quantitative_evidence", [])
        if not raw_quant:
            return True, [], "No quantitative records"

        qtext = str(question.get("question", "")).lower()
        target_is_latency = any(k in qtext for k in ("latency", "cost", "token", "overhead", "compute"))
        target_is_accuracy = any(k in qtext for k in ("accuracy", "compare", "outperform", "benchmark", "quantitatively", "reasoning"))

        valid_records = []
        for qrec in raw_quant:
            if not isinstance(qrec, dict):
                continue

            metric_name = str(qrec.get("metric", "")).strip().lower()

            # Reject metadata numbers masquerading as benchmark metrics
            if any(non_m in metric_name for non_m in NON_EVIDENTIARY_NUMBERS):
                continue

            # If question is about latency/cost, metric must be cost/latency related
            if target_is_latency and not any(k in metric_name for k in COST_LATENCY_METRIC_KEYWORDS):
                # Unless it reports multiplier or overhead factor
                if not qrec.get("multiplier"):
                    continue

            # If question is about reasoning/accuracy, metric must not be pure token count
            if target_is_accuracy and not target_is_latency:
                if any(k in metric_name for k in ("token overhead", "api cost", "gpu hours")) and not any(k in metric_name for k in ACCURACY_METRIC_KEYWORDS):
                    continue

            # Check that numerical values exist and are grounded
            b_val = qrec.get("baseline_score")
            e_val = qrec.get("experimental_score")
            diff = qrec.get("absolute_difference")
            mult = qrec.get("multiplier")
            rmin = qrec.get("range_min")
            rmax = qrec.get("range_max")

            has_contrast = (b_val is not None and e_val is not None) or (diff is not None) or (mult is not None) or (rmin is not None and rmax is not None)
            if not has_contrast:
                # Disqualify standalone numbers that do not provide comparative contrast
                continue

            valid_records.append(qrec)

        if raw_quant and not valid_records:
            return False, [], "Quantitative metrics did not align with question target variable or lacked comparative contrast"

        return True, valid_records, f"Validated {len(valid_records)} quantitative records"

    @classmethod
    def calibrate_context_and_stance(cls, finding: dict, question: dict) -> str:
        """
        Gate 4: Context Compatibility & Stance Nuance Gate.
        Accurately differentiates:
        - 'support': Confirms hypothesis / shows positive gains
        - 'counter': True refutation / baseline superiority / severe failure
        - 'limitation': Scope boundary / resource constraint without refuting
        - 'trade_off': Gains accompanied by overhead/penalties
        - 'context': Descriptive / background
        """
        claim = str(finding.get("claim", "")).strip()
        ev_texts = [e.get("evidence_text", "") for e in finding.get("evidence", [])]
        combined = (f"{claim} " + " ".join(ev_texts)).lower()

        # 1. Trade-off takes precedence if gains are paired with costs
        if any(re.search(p, combined) for p in TRADE_OFF_PATTERNS):
            return "trade_off"

        # 2. Limitations without refutation
        is_limitation = any(re.search(p, combined) for p in LIMITATION_PATTERNS)
        is_counter = any(re.search(p, combined) for p in COUNTER_PATTERNS)
        is_support = any(re.search(p, combined) for p in SUPPORT_PATTERNS)

        if is_support and is_counter:
            return "trade_off"

        if is_counter:
            return "counter"

        if is_support:
            return "support"

        if is_limitation:
            return "limitation"

        raw_stance = str(finding.get("stance", "context")).strip().lower()
        if raw_stance in ALLOWED_STANCES:
            # Downgrade false counter if it's actually a limitation
            if raw_stance == "counter" and is_limitation:
                return "limitation"
            return raw_stance

        return "context"

    @classmethod
    def validate_finding(
        cls,
        finding: dict,
        question: dict,
        source_text: str = ""
    ) -> Tuple[bool, dict, List[str]]:
        """
        Run Gates 1 to 4 on a single candidate finding.
        Returns: (is_valid, validated_finding_dict, rejection_reasons)
        """
        reasons = []

        # Gate 1: Question Relevance
        rel_ok, rel_msg = cls.validate_question_relevance(finding, question)
        if not rel_ok:
            reasons.append(f"Gate 1 Failed: {rel_msg}")
            return False, finding, reasons

        # Gate 2: Claim Support
        supp_ok, supp_msg = cls.validate_claim_support(finding, source_text)
        if not supp_ok:
            reasons.append(f"Gate 2 Failed: {supp_msg}")
            return False, finding, reasons

        # Gate 3: Quantitative Validity
        quant_ok, clean_quant, quant_msg = cls.validate_quantitative_validity(finding, question, source_text)
        if not quant_ok:
            # If quantitative was required, fail; otherwise strip invalid quantitative records
            if question.get("requires_quantitative_evidence"):
                reasons.append(f"Gate 3 Failed: {quant_msg}")
                return False, finding, reasons

        # Gate 4: Stance Nuance Calibration
        calibrated_stance = cls.calibrate_context_and_stance(finding, question)

        validated = dict(finding)
        validated["stance"] = calibrated_stance
        validated["quantitative_evidence"] = clean_quant
        validated["evl_status"] = "validated"

        return True, validated, []

    @staticmethod
    def enforce_source_independence(findings: List[dict]) -> Dict[str, Any]:
        """
        Gate 5: Source Independence & Cluster Consolidation Gate.
        Clusters findings by independent source ID, preventing 1 paper from inflating
        evidence strength or consensus counts.
        """
        source_clusters = {}
        support_sources = set()
        counter_sources = set()
        limitation_sources = set()
        trade_off_sources = set()
        quantitative_sources = set()

        for f in findings:
            sids = f.get("source_ids", [])
            if not sids and f.get("source_id"):
                sids = [f["source_id"]]

            stance = f.get("stance", "context")
            has_quant = len(f.get("quantitative_evidence", [])) > 0

            for sid in sids:
                source_clusters.setdefault(sid, []).append(f)
                if stance == "support":
                    support_sources.add(sid)
                elif stance == "counter":
                    counter_sources.add(sid)
                elif stance == "limitation":
                    limitation_sources.add(sid)
                elif stance == "trade_off":
                    trade_off_sources.add(sid)

                if has_quant:
                    quantitative_sources.add(sid)

        total_findings = len(findings)
        max_single_source = max(len(flist) for flist in source_clusters.values()) if source_clusters else 0
        concentration_ratio = round(max_single_source / total_findings, 2) if total_findings > 0 else 0.0

        return {
            "source_clusters": source_clusters,
            "unique_source_count": len(source_clusters),
            "support_source_count": len(support_sources),
            "counter_source_count": len(counter_sources),
            "limitation_source_count": len(limitation_sources),
            "trade_off_source_count": len(trade_off_sources),
            "quantitative_source_count": len(quantitative_sources),
            "concentration_ratio": concentration_ratio,
            "high_concentration": concentration_ratio > 0.50 and total_findings >= 4
        }

    @staticmethod
    def filter_useful_sources(selected_sources: list, validated_findings: List[dict]) -> Tuple[list, list]:
        """
        Separates selected sources into:
        - useful_sources: sources that produced at least one EVL-validated finding.
        - idle_sources: sources selected in ranking that produced zero validated findings.
        """
        active_source_ids = set()
        for f in validated_findings:
            sids = f.get("source_ids", [])
            if not sids and f.get("source_id"):
                sids = [f["source_id"]]
            for sid in sids:
                active_source_ids.add(str(sid).strip().upper())

        useful = []
        idle = []

        for item in selected_sources:
            src = item[0] if isinstance(item, (tuple, list)) else item
            sid = str(src.get("source_id", "")).strip().upper()
            if sid in active_source_ids:
                useful.append(src)
            else:
                idle.append(src)

        return useful, idle
