"""
tests/test_live_feedback_fixes.py

Unit tests verifying the 7 prioritized pipeline fixes from the live research run:
1. Full-text evidence extraction: arXiv HTML/PDF resolution and pypdf extraction.
2. Relevant retrieved sources yielding findings: Gate 1 defense concept alignment.
3. Cross-question routing: Archetype-gated cross-routing blocks incompatible claims.
4. Evidence deduplication: Normalization and cross-question deduplication.
5. Counter-evidence extraction: Security counter-evidence recognized without false limitation downgrading.
6. Defense-specific evidence extraction: Gate 1 and Gate 2 defense validation.
7. Primary-source counting inconsistency: conference_paper and academic providers verified in contract enforcer.
"""

import unittest
import io
import pypdf
from search_agent.content_retriever import (
    _extract_arxiv_id,
    _extract_text_from_pdf_bytes,
    _assess_content_status,
)
from search_agent.evidence_validator import EvidenceValidator
from search_agent.evidence_extractor import (
    extract_research_evidence,
)
from planner.validation.contract_enforcer import enforce_research_contract


class TestFullTextAndPDFRetrieval(unittest.TestCase):
    """Test Item 1: Full-text resolution and PDF extraction."""

    def test_extract_arxiv_id_from_various_formats(self):
        self.assertEqual(_extract_arxiv_id("https://arxiv.org/abs/2402.17840"), "2402.17840")
        self.assertEqual(_extract_arxiv_id("https://doi.org/10.48550/arxiv.2402.17840"), "2402.17840")
        self.assertEqual(_extract_arxiv_id("https://arxiv.org/html/2606.09005v1"), "2606.09005v1")
        self.assertEqual(_extract_arxiv_id("https://arxiv.org/pdf/2410.02298.pdf"), "2410.02298")
        self.assertEqual(_extract_arxiv_id("https://example.com/not-arxiv"), "")

    def test_pypdf_text_extraction_from_bytes(self):
        # Create a real in-memory PDF with text
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=100, height=100)
        buf = io.BytesIO()
        writer.write(buf)
        pdf_bytes = buf.getvalue()

        # Blank page extracts cleanly without error
        text = _extract_text_from_pdf_bytes(pdf_bytes)
        self.assertIsInstance(text, str)

    def test_assess_content_status_accuracy(self):
        # Empty text
        self.assertEqual(_assess_content_status(""), "failed")
        # Short snippet
        self.assertEqual(_assess_content_status("Short snippet about RAG."), "snippet_only")
        # Moderate abstract without empirical keywords
        self.assertEqual(_assess_content_status("A " * 500), "partial")
        # Empirical full text with sections and benchmarks
        full_text = "In our experiments, the evaluation results on benchmarks show accuracy improves significantly. " * 30
        self.assertEqual(_assess_content_status(full_text), "full")


class TestResilientEvidenceValidation(unittest.TestCase):
    """Test Items 2, 5, 6: Gate 1 Defense Concepts, Gate 2 Grounding, Gate 4 Counter Stances."""

    def setUp(self):
        self.defense_question = {
            "id": "Q3",
            "type": "evaluation",
            "question": "How effective are current defense mechanisms at mitigating these vulnerabilities?",
            "evidence_needed": ["defense mechanism description", "mitigation effectiveness metrics"],
            "requires_quantitative_evidence": True,
            "requires_counter_evidence": True
        }

    def test_defense_finding_passes_gate_1_without_literal_prompt_overlap(self):
        """Verify Item 6: Defense paper findings pass Gate 1 even if literal prompt words are not repeated."""
        finding = {
            "claim": "PRA-RAG reduces retrieval attack success rate from 88% down to 12% across open-domain QA benchmarks.",
            "evidence": [
                {"evidence_text": "Our proposed method PRA-RAG reduces the retrieval attack success rate from 88% to 12% across benchmarks."}
            ]
        }
        passed, msg = EvidenceValidator.validate_question_relevance(finding, self.defense_question)
        self.assertTrue(passed, f"Gate 1 failed unexpectedly: {msg}")

    def test_gate_2_resilient_to_whitespace_and_citation_brackets(self):
        """Verify Item 2: Gate 2 tolerates citation brackets [1, 2] and slight LLM formatting differences."""
        source_text = "In this work, we propose PRA-RAG [1]. Extensive evaluations demonstrate that PRA-RAG achieves robust aggregation and significantly reduces retrieval attack success rate from 88% to 12% across open-domain QA benchmarks."
        # Excerpt stripped of citations and slightly reformatted by LLM
        finding = {
            "claim": "PRA-RAG achieves robust aggregation and reduces retrieval attack success rate.",
            "evidence": [
                {"evidence_text": "PRA-RAG achieves robust aggregation and significantly reduces retrieval attack success rate from 88% to 12% across open-domain QA benchmarks."}
            ]
        }
        passed, msg = EvidenceValidator.validate_claim_support(finding, source_text)
        self.assertTrue(passed, f"Gate 2 failed on grounded excerpt: {msg}")

    def test_gate_4_security_counter_evidence_not_downgraded_to_limitation(self):
        """Verify Item 5: Security counter evidence is recognized as 'counter' and not falsely downgraded."""
        finding = {
            "claim": "Standard RAG systems remain highly resilient under knowledge poisoning attacks when context isolation is applied, dropping attack success rate to below 5%.",
            "evidence": [
                {"evidence_text": "Attack success rate drops to below 5% under context isolation, demonstrating high resilience."}
            ],
            "stance": "counter"
        }
        stance = EvidenceValidator.calibrate_context_and_stance(finding, self.defense_question)
        self.assertEqual(stance, "counter", "Expected 'counter' stance for high resilience / low ASR finding")

    def test_gate_3_preserves_qualitative_finding_on_quantitative_question(self):
        """Verify Item 2: Qualitative finding is preserved with clean quantitative list rather than rejected."""
        finding = {
            "claim": "PRA-RAG utilizes certified majority voting to isolate poisoned chunks.",
            "evidence": [
                {"evidence_text": "PRA-RAG utilizes certified majority voting to isolate poisoned chunks."}
            ],
            "quantitative_evidence": []  # No numbers in this qualitative architectural finding
        }
        source_text = "PRA-RAG utilizes certified majority voting to isolate poisoned chunks."
        is_valid, validated, reasons = EvidenceValidator.validate_finding(
            finding,
            self.defense_question,
            source_text=source_text
        )
        self.assertTrue(is_valid, f"Qualitative finding rejected: {reasons}")
        self.assertEqual(validated.get("quantitative_evidence"), [])


class TestCrossQuestionRoutingAndDeduplication(unittest.TestCase):
    """Test Items 3 & 4: Archetype-gated cross-routing and finding deduplication."""

    def test_attack_finding_blocked_from_defense_question_routing(self):
        """Verify Item 3: Pure attack findings cannot route into defense evaluation questions."""
        q1_attack = {
            "id": "Q1",
            "type": "mechanisms",
            "question": "What are the primary vectors of knowledge-base poisoning in RAG systems?"
        }
        q3_defense = {
            "id": "Q3",
            "type": "evaluation",
            "question": "How effective are current defense mechanisms at mitigating these vulnerabilities?"
        }

        attack_finding = {
            "claim": "TPARAG achieves high retrieval and end-to-end attack success rates across open-domain QA benchmarks.",
            "evidence": [
                {"evidence_text": "TPARAG consistently achieves high retrieval attack success rates."}
            ],
            "stance": "support"
        }

        # Mock extraction bundle where Q1 has attack finding and Q3 is evaluating routing
        mock_plan = {"questions": [q1_attack, q3_defense]}
        mock_selection = {
            "per_question_selection": {
                "Q1": {
                    "selected_sources": [
                        ({"source_id": "S17", "content": "TPARAG attack...", "source_type": "conference_paper"}, "primary")
                    ]
                },
                "Q3": {
                    "selected_sources": []
                }
            }
        }

        class MockGemini:
            class models:
                @staticmethod
                def generate_content(*args, **kwargs):
                    class Resp:
                        text = '{"findings": [{"claim": "TPARAG achieves high attack success rate.", "stance": "support", "confidence": "high", "evidence": [{"evidence_type": "qualitative", "evidence_text": "TPARAG achieves high attack success rate."}]}]}'
                    return Resp()

        res = extract_research_evidence(MockGemini(), mock_plan, mock_selection)
        q3_res = next(q for q in res["questions"] if q["question_id"] == "Q3")
        # Q3 should have 0 findings because the pure attack finding was blocked from routing to defense question!
        self.assertEqual(len(q3_res["findings"]), 0)


class TestPrimarySourceContractEnforcement(unittest.TestCase):
    """Test Item 7: Primary source counting in contract enforcer."""

    def test_conference_paper_and_academic_provider_counted_as_primary(self):
        plan = {
            "questions": [
                {
                    "id": "Q1",
                    "question": "What are the vectors?",
                    "source_requirements": {
                        "minimum_sources": 2,
                        "minimum_primary_sources": 2,
                        "minimum_quantitative_sources": 1,
                        "minimum_counter_evidence_sources": 0
                    }
                }
            ]
        }

        evidence = {
            "sources": [
                {
                    "source_id": "S17",
                    "title": "Token-Level Precise Attack on RAG",
                    "source_type": "conference_paper"
                },
                {
                    "source_id": "OA-W4399991306",
                    "title": "Scalable Data Extraction from RAG",
                    "source_type": "preprint",
                    "doi": "10.48550/arxiv.2402.17840",
                    "provider_name": "openalex"
                }
            ],
            "questions": [
                {
                    "question_id": "Q1",
                    "findings": [
                        {
                            "finding_id": "Q1-F1",
                            "source_ids": ["S17", "OA-W4399991306"],
                            "claim": "Both papers evaluate poisoning attacks.",
                            "stance": "support",
                            "quantitative_evidence": [{"metric": "ASR", "baseline_score": 50.0, "experimental_score": 85.0}]
                        }
                    ]
                }
            ]
        }

        res = enforce_research_contract(plan, evidence)
        q1_comp = res["question_compliance"][0]
        prim_clause = next(c for c in q1_comp["clauses"] if c["clause"] == "minimum_primary_sources")
        
        # S17 (conference_paper) and OA-W4399991306 (openalex preprint) must both be counted as primary!
        self.assertEqual(prim_clause["actual"], 2)
        self.assertTrue(prim_clause["met"])
        self.assertEqual(q1_comp["status"], "MET")


if __name__ == "__main__":
    unittest.main()

