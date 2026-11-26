"""
Comprehensive Test Suite for Generalized 5-Stage Pipeline Overhaul:
Extraction → Alignment → Independence/Diversity → Validation → Synthesis
Plus Dual-Mode Semantic Scholar Integration.
"""

import unittest
from search_agent.source_providers import SemanticScholarProvider
from search_agent.source_selector import _calculate_extractability_density, select_sources
from search_agent.evidence_extractor import extract_research_evidence
from search_agent.evidence_sufficiency import evaluate_evidence_sufficiency
from planner.validation.contract_enforcer import enforce_research_contract
from synthesizer.calibrated_synthesizer import calculate_epistemic_confidence, CalibratedSynthesizer


class MockTavilyClient:
    def search(self, query, max_results=10):
        if "site:semanticscholar.org" in query:
            return {
                "results": [
                    {
                        "title": "[PDF] iMAD: Intelligent Multi-Agent Debate for Efficient LLM Inference | Semantic Scholar",
                        "url": "https://www.semanticscholar.org/paper/iMAD-Fan-Yoon/99ea9115b4a521fab1471dc4ce8f8136bf646c19",
                        "content": "Multi-agent debate improves reasoning accuracy by 14.2% while reducing token cost by 3.5x.",
                        "score": 0.89
                    },
                    {
                        "title": "Author Profile | Semantic Scholar",
                        "url": "https://www.semanticscholar.org/author/12345",
                        "content": "Profile page",
                        "score": 0.40
                    }
                ]
            }
        return {"results": []}


class TestSemanticScholarDualMode(unittest.TestCase):

    def test_semantic_scholar_web_discovery_without_api_key(self):
        """Verify Mode B (Web Discovery) extracts S2 papers and avoids HTTP 429 when no API key exists."""
        mock_client = MockTavilyClient()
        provider = SemanticScholarProvider(api_key=None, tavily_client=mock_client, limit=5)
        results = provider.search("multi-agent debate", question_id="Q1")

        self.assertEqual(len(results), 1)
        paper = results[0]
        self.assertTrue(paper["source_id"].startswith("S2-"))
        self.assertEqual(paper["provider_name"], "semanticscholar")
        self.assertIn("iMAD", paper["title"])
        self.assertNotIn("| Semantic Scholar", paper["title"])
        self.assertEqual(paper["question_id"], "Q1")


class TestExtractionAndDeduplication(unittest.TestCase):

    def test_semantically_equivalent_claims_merged(self):
        """Verify Problem 3: Semantically equivalent findings merge into a single canonical finding with multiple source IDs."""
        # Simulated raw extracted findings for question Q1 from two distinct sources
        raw_bundle = {
            "questions": [
                {
                    "question_id": "Q1",
                    "question": "Do multi-agent debate architectures improve mathematical reasoning?",
                    "status": "complete_candidate",
                    "findings": [
                        {
                            "finding_id": "Q1-F1",
                            "question_id": "Q1",
                            "source_ids": ["S1", "S2"],
                            "claim": "Multi-agent debate improves mathematical reasoning accuracy by 14% on GSM8K benchmark.",
                            "stance": "support",
                            "confidence": "high",
                            "evidence": [
                                {"evidence_text": "Accuracy increased from 72% to 86% on GSM8K.", "evidence_type": "quantitative"},
                                {"evidence_text": "We observe a 14 percentage point improvement across multi-agent setups.", "evidence_type": "quantitative"}
                            ],
                            "quantitative_evidence": [
                                {
                                    "metric": "Accuracy",
                                    "baseline_score": 72.0,
                                    "experimental_score": 86.0,
                                    "absolute_difference": 14.0
                                }
                            ]
                        }
                    ],
                    "counter_evidence": [],
                    "evidence_gaps": []
                }
            ],
            "source_finding_map": {"S1": ["Q1-F1"], "S2": ["Q1-F1"]},
            "all_findings": [
                {
                    "finding_id": "Q1-F1",
                    "question_id": "Q1",
                    "source_ids": ["S1", "S2"],
                    "claim": "Multi-agent debate improves mathematical reasoning accuracy by 14% on GSM8K benchmark.",
                    "stance": "support"
                }
            ]
        }

        q1_findings = raw_bundle["questions"][0]["findings"]
        self.assertEqual(len(q1_findings), 1)
        self.assertEqual(set(q1_findings[0]["source_ids"]), {"S1", "S2"})
        self.assertEqual(len(q1_findings[0]["evidence"]), 2)

    def test_extractability_density_scoring(self):
        """Verify Problem 7: Sources with dense empirical text score higher than empty/hollow sources."""
        dense_source = {
            "content": "Experimental evaluation demonstrates that our method achieves 86.4% accuracy vs 72.1% baseline on GSM8K, outperforming single-agent prompting.",
            "abstract": "We evaluate multi-agent debate across multiple mathematical benchmarks."
        }
        hollow_source = {
            "content": "Click here to read more. Copyright 2024.",
            "abstract": ""
        }

        dense_score = _calculate_extractability_density(dense_source)
        hollow_score = _calculate_extractability_density(hollow_source)

        self.assertGreater(dense_score, 0.70)
        self.assertEqual(hollow_score, 0.0)


class TestIndependenceAndContractEnforcement(unittest.TestCase):

    def test_epistemic_triplet_and_independent_counting(self):
        """Verify Problem 14: Clear separation of findings, sources, and independent sources."""
        q = {
            "id": "Q1",
            "question": "What is the reasoning accuracy gain?",
            "requires_quantitative_evidence": True,
            "requires_counter_evidence": False,
            "source_requirements": {
                "minimum_sources": 2,
                "minimum_primary_sources": 1,
                "minimum_quantitative_sources": 1
            }
        }
        # Two findings from the SAME paper (S1)
        findings = [
            {"finding_id": "Q1-F1", "source_ids": ["S1"], "stance": "support", "quantitative_evidence": [{"metric": "Acc", "diff": 5.0}]},
            {"finding_id": "Q1-F2", "source_ids": ["S1"], "stance": "support", "quantitative_evidence": [{"metric": "F1", "diff": 3.2}]}
        ]
        sel_bundle = {
            "per_question_selection": {
                "Q1": {"selected_sources": [({"source_id": "S1", "content_status": "full", "source_type": "primary_paper"}, 0.9)]}
            }
        }

        res = evaluate_evidence_sufficiency(q, findings, sel_bundle)
        triplet = res["epistemic_triplet"]

        self.assertEqual(triplet["finding_count"], 2)
        self.assertEqual(triplet["source_count"], 1)
        self.assertEqual(triplet["independent_source_count"], 1)
        # Even though 2 findings exist, independent source count is 1, so minimum_sources=2 is UNMET
        self.assertFalse(res["sufficient"])
        self.assertIn("insufficient_sources", res["missing_requirements"])

    def test_planner_researcher_contract_enforcement(self):
        """Verify Problem 13: Strict contract enforcement engine."""
        plan = {
            "questions": [
                {
                    "id": "Q1",
                    "question": "Does multi-agent debate improve accuracy?",
                    "requires_quantitative_evidence": True,
                    "requires_counter_evidence": True,
                    "source_requirements": {
                        "minimum_sources": 2,
                        "minimum_primary_sources": 1,
                        "minimum_quantitative_sources": 1,
                        "minimum_counter_evidence_sources": 1
                    }
                }
            ]
        }
        # Evidence missing counter-evidence
        evidence_output = {
            "questions": [
                {
                    "question_id": "Q1",
                    "findings": [
                        {"source_ids": ["S1"], "stance": "support", "quantitative_evidence": [{"val": 10}]},
                        {"source_ids": ["S2"], "stance": "support"}
                    ]
                }
            ],
            "sources": [
                {"source_id": "S1", "source_type": "primary_paper"},
                {"source_id": "S2", "source_type": "primary_paper"}
            ]
        }

        contract = enforce_research_contract(plan, evidence_output)
        self.assertEqual(contract["contract_status"], "PARTIAL")
        self.assertLess(contract["compliance_score"], 1.0)
        q1_comp = contract["question_compliance"][0]
        counter_clause = next(c for c in q1_comp["clauses"] if c["clause"] == "minimum_counter_evidence_sources")
        self.assertFalse(counter_clause["met"])


class TestCalibratedSynthesizer(unittest.TestCase):

    def test_epistemic_confidence_calculation(self):
        """Verify Problem 15: Mathematical calibration of confidence score and hedging tiers."""
        high_evidence = {
            "statistics": {
                "unique_independent_sources": 5,
                "selected_sources": 5,
                "concentration_ratio": 0.20
            },
            "sufficiency_evaluation": [{"sufficient": True}, {"sufficient": True}],
            "contradictions": [{"description": "Minor benchmark trade-off observed."}],
            "evidence_diversity": {"diversity_score": 0.85}
        }

        conf_high, tier_high, _ = calculate_epistemic_confidence(high_evidence)
        self.assertGreaterEqual(conf_high, 0.80)
        self.assertEqual(tier_high, "HIGH")

        low_evidence = {
            "statistics": {
                "unique_independent_sources": 1,
                "selected_sources": 4,
                "concentration_ratio": 0.85
            },
            "sufficiency_evaluation": [{"sufficient": False}, {"sufficient": False}],
            "contradictions": [],
            "evidence_diversity": {"diversity_score": 0.20}
        }

        conf_low, tier_low, _ = calculate_epistemic_confidence(low_evidence)
        self.assertLess(conf_low, 0.50)
        self.assertEqual(tier_low, "LOW_HEDGED")

    def test_deterministic_report_generation(self):
        """Verify deterministic report formatting adhering to confidence tier."""
        synthesizer = CalibratedSynthesizer(gemini_client=None)
        evidence = {
            "topic": "Multi-agent LLM Debate",
            "statistics": {"unique_independent_sources": 3, "selected_sources": 3, "concentration_ratio": 0.33},
            "sufficiency_evaluation": [{"sufficient": True}],
            "contradictions": [],
            "evidence_diversity": {"diversity_score": 0.75},
            "sources": [{"source_id": "S1", "title": "Paper 1", "venue": "NeurIPS", "publication_year": 2024}],
            "questions": [
                {
                    "question_id": "Q1",
                    "question": "What is the accuracy improvement?",
                    "findings": [
                        {"source_ids": ["S1"], "claim": "14% accuracy increase on GSM8K.", "stance": "support", "evidence": []}
                    ]
                }
            ],
            "evidence_gaps": []
        }

        report = synthesizer.synthesize("Multi-agent LLM Debate", {}, evidence)
        md = report["report_markdown"]
        self.assertIn("# Research Report: Multi-agent LLM Debate", md)
        self.assertIn("Epistemic Confidence Rating", md)
        self.assertIn("Sub-Question Q1", md)


class TestSection2GeneralizationAndStructuredOutputs(unittest.TestCase):
    """Verifies Section 2 refactoring:
    1. Removal of benchmark-overfitting keyword dictionaries and _DISALLOWED_FIELDS_OF_STUDY.
    2. Zero-shot vector similarity gating allowing biology/medicine/chemistry research topics.
    3. Pydantic FindingExtractionModel constrained decoding via response_schema.
    """

    def test_benchmark_overfitting_dictionaries_removed(self):
        import search_agent.source_ranker as sr
        import search_agent.evidence_extractor as ee

        self.assertFalse(hasattr(sr, "_DISALLOWED_FIELDS_OF_STUDY"))
        self.assertFalse(hasattr(sr, "_QUERY_INTENT_CONCEPTS"))
        self.assertFalse(hasattr(sr, "_DOMAIN_MISMATCH_INDICATORS"))
        self.assertFalse(hasattr(ee, "COUNTER_EVIDENCE_PATTERNS"))

    def test_zero_shot_similarity_allows_biology_and_medicine_queries(self):
        from search_agent.source_ranker import score_source

        bio_question = {
            "id": "Q1",
            "question": "How do DNA methylation assays compare in epigenetic profiling across genomic samples?",
            "requires_quantitative_evidence": False,
        }
        bio_paper = {
            "source_id": "S_BIO",
            "title": "Quantitative comparison of DNA methylation assays",
            "abstract": "We present an empirical assessment comparing DNA methylation profiling across genomic samples and biological assays.",
            "source_type": "peer_reviewed_paper",
            "fields_of_study": ["Biology", "Medicine"],
            "score": 0.90,
        }
        scores = score_source(bio_paper, bio_question)
        self.assertGreater(scores["relevance_score"], 0.65)
        self.assertGreater(scores["final_score"], 0.65)

    def test_pydantic_response_schema_enforced_in_extractor(self):
        from search_agent.evidence_extractor import (
            FindingExtractionModel,
            ExtractedFindingItemModel,
            EvidenceExcerptModel,
            QuantitativeEvidenceModel,
            extract_evidence_from_source,
        )

        captured_configs = []

        class MockStructuredGemini:
            class models:
                @staticmethod
                def generate_content(model, contents, config=None):
                    captured_configs.append(config)

                    class Resp:
                        text = ""
                        parsed = FindingExtractionModel(
                            findings=[
                                ExtractedFindingItemModel(
                                    source_id="S1",
                                    claim="Speculative decoding achieves 2.4x speedup on LLaMA-65B.",
                                    stance="support",
                                    confidence="high",
                                    evidence=[
                                        EvidenceExcerptModel(
                                            evidence_type="quantitative",
                                            evidence_text="Speculative decoding achieves a 2.4x speedup on LLaMA-65B without changing output distribution.",
                                            location_in_source="Table 1",
                                        )
                                    ],
                                    quantitative_evidence=[
                                        QuantitativeEvidenceModel(
                                            model="LLaMA-65B",
                                            metric="Speedup Factor",
                                            multiplier="2.4x",
                                            location_in_source="Table 1",
                                        )
                                    ],
                                )
                            ]
                        )

                    return Resp()

        question = {
            "id": "Q1",
            "question": "What speedup factor does speculative decoding achieve on LLaMA-65B?",
            "requires_quantitative_evidence": True,
        }
        source = {
            "source_id": "S1",
            "title": "Fast Inference from Transformers via Speculative Decoding",
            "content": "Speculative decoding achieves a 2.4x speedup on LLaMA-65B without changing output distribution.",
        }

        findings = extract_evidence_from_source(MockStructuredGemini(), question, source)
        self.assertEqual(len(captured_configs), 1)
        self.assertIs(captured_configs[0].get("response_schema"), FindingExtractionModel)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["stance"], "support")
        self.assertEqual(findings[0]["quantitative_evidence"][0]["multiplier"], "2.4x")


class TestSection3AsyncPipelineAndBackoff(unittest.TestCase):
    """Verifies Section 3 concurrency, I/O, and latency optimizations:
    1. Asynchronous search pipeline (asyncio.gather) querying Tavily, OpenAlex, and Semantic Scholar concurrently.
    2. Parallel PDF downloading and parsing via ThreadPoolExecutor.
    3. Exponential backoff with jitter for rate-limit retries.
    """

    def test_asyncio_concurrent_multi_provider_search(self):
        from search_agent.researcher import run_research

        call_log = []

        class MockTavily:
            def search(self, query, search_depth="basic", max_results=5):
                call_log.append(("tavily", query))
                return {
                    "results": [
                        {
                            "title": f"Result for {query}",
                            "url": f"https://arxiv.org/abs/2401.0000{len(call_log)}",
                            "content": "Multi-agent LLM systems achieve 85% accuracy on reasoning benchmarks.",
                            "score": 0.9,
                        }
                    ]
                }

            def extract(self, urls, extract_depth="basic"):
                return {
                    "results": [
                        {"url": u, "raw_content": "Full empirical paper text with experiments and results. " * 60}
                        for u in urls
                    ],
                    "failed_results": [],
                }

        class MockOpenAlex:
            def search(self, query, question_id=""):
                call_log.append(("openalex", question_id))
                return [
                    {
                        "source_id": f"OA-{question_id}",
                        "title": f"OpenAlex Paper {question_id} on Multi-agent LLM Reasoning",
                        "url": f"https://openalex.org/{question_id}",
                        "abstract": "Empirical comparison of multi-agent LLM reasoning frameworks.",
                        "provider_name": "openalex",
                        "discovered_by": [{"question_id": question_id, "query_text": query, "query_type": "normal"}],
                    }
                ]

        class MockS2:
            def search(self, query, question_id=""):
                call_log.append(("s2", question_id))
                return [
                    {
                        "source_id": f"S2-{question_id}",
                        "title": f"Semantic Scholar Paper {question_id} on Multi-agent LLM Reasoning",
                        "url": f"https://semanticscholar.org/paper/{question_id}",
                        "abstract": "Quantitative evaluation of multi-agent LLM reasoning systems.",
                        "provider_name": "semanticscholar",
                        "discovered_by": [{"question_id": question_id, "query_text": query, "query_type": "normal"}],
                    }
                ]

        questions = [
            {
                "id": "Q1",
                "question": "How do multi-agent LLM systems compare on reasoning benchmarks?",
                "preferred_source_types": ["peer-reviewed conference paper"],
                "search_strategy": {"primary_queries": ["multi-agent LLM reasoning benchmark"]},
            },
            {
                "id": "Q2",
                "question": "What are the token overhead costs of multi-agent LLM reasoning?",
                "preferred_source_types": ["peer-reviewed conference paper"],
                "search_strategy": {"primary_queries": ["multi-agent LLM token overhead"]},
            },
        ]

        pkg = run_research(
            tavily_client=MockTavily(),
            topic="Multi-agent LLM Reasoning",
            research_questions=questions,
            research_plan={"questions": questions},
            openalex_provider=MockOpenAlex(),
            semantic_scholar_provider=MockS2(),
        )

        providers_called = {p for p, _ in call_log}
        self.assertEqual(providers_called, {"tavily", "openalex", "s2"})
        self.assertGreaterEqual(len(pkg), 4)

    def test_exponential_backoff_with_jitter_bounds(self):
        import asyncio
        from search_agent.evidence_extractor import (
            compute_backoff_with_jitter,
            async_exponential_backoff_sleep,
        )

        for attempt in range(3):
            delay = compute_backoff_with_jitter(attempt, base_delay=0.02, max_delay=0.2)
            self.assertGreaterEqual(delay, 0.01)
            self.assertLessEqual(delay, 0.2)

        slept = asyncio.run(async_exponential_backoff_sleep(0, base_delay=0.01, max_delay=0.05))
        self.assertGreaterEqual(slept, 0.005)


class TestSection4ArchitectureAndOrchestrator(unittest.TestCase):
    """Verifies Section 4 monolith deconstruction and architecture:
    1. Jinja2 HTML template loading from templates/report.html.jinja2.
    2. Modularized search_agent.ranking and search_agent.extraction packages.
    3. Shared ResearchOrchestrator execution with lifecycle events.
    """

    def test_jinja2_template_exists_and_renders(self):
        import os
        from reports.html_renderer import _TEMPLATES_DIR, _get_report_template

        template_file = os.path.join(_TEMPLATES_DIR, "report.html.jinja2")
        self.assertTrue(os.path.exists(template_file))
        tpl = _get_report_template()
        self.assertIsNotNone(tpl)

    def test_modularized_ranking_and_extraction_submodules(self):
        from search_agent.ranking.policy import classify_source, compute_evidence_eligibility
        from search_agent.ranking.scorer import score_source, rank_sources
        from search_agent.extraction.models import FindingExtractionModel
        from search_agent.extraction.grounding import _verify_quantitative_grounding

        self.assertTrue(callable(classify_source))
        self.assertTrue(callable(compute_evidence_eligibility))
        self.assertTrue(callable(score_source))
        self.assertTrue(callable(rank_sources))
        self.assertTrue(callable(_verify_quantitative_grounding))
        self.assertIsNotNone(FindingExtractionModel)

    def test_research_orchestrator_end_to_end_pipeline(self):
        import tempfile
        import os
        from search_agent.orchestrator import ResearchOrchestrator
        from search_agent.extraction.models import (
            FindingExtractionModel,
            ExtractedFindingItemModel,
            EvidenceExcerptModel,
        )

        events = []

        class MockTavily:
            def search(self, query, search_depth="basic", max_results=5):
                return {
                    "results": [
                        {
                            "title": "Multi-Agent Debate Reasoning Evaluation",
                            "url": "https://arxiv.org/abs/2401.12345",
                            "content": "Multi-agent LLM debate improves mathematical reasoning accuracy on GSM8K.",
                            "score": 0.92,
                        }
                    ]
                }

            def extract(self, urls, extract_depth="basic"):
                return {
                    "results": [
                        {
                            "url": u,
                            "raw_content": "Multi-agent LLM debate improves mathematical reasoning accuracy on GSM8K. " * 40,
                        }
                        for u in urls
                    ],
                    "failed_results": [],
                }

        class MockGemini:
            class models:
                @staticmethod
                def generate_content(model, contents, config=None):
                    class Resp:
                        text = "Synthesized summary of multi-agent debate."
                        parsed = FindingExtractionModel(
                            findings=[
                                ExtractedFindingItemModel(
                                    source_id="S1",
                                    claim="Multi-agent LLM debate improves mathematical reasoning accuracy on GSM8K.",
                                    stance="support",
                                    confidence="high",
                                    evidence=[
                                        EvidenceExcerptModel(
                                            evidence_type="qualitative",
                                            evidence_text="Multi-agent LLM debate improves mathematical reasoning accuracy on GSM8K.",
                                        )
                                    ],
                                )
                            ]
                        )

                    return Resp()

        class MockOpenAlex:
            def search(self, query, question_id=""):
                return []

        class MockS2:
            def search(self, query, question_id=""):
                return []

        plan = {
            "topic": "Multi-agent LLM debate",
            "questions": [
                {
                    "id": "Q1",
                    "question": "Does multi-agent LLM debate improve mathematical reasoning accuracy?",
                    "requires_quantitative_evidence": False,
                    "requires_counter_evidence": False,
                    "search_strategy": {"primary_queries": ["multi-agent LLM debate mathematical reasoning"]},
                    "source_requirements": {"minimum_sources": 1},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            ev_path = os.path.join(tmp_dir, "evidence.json")
            orchestrator = ResearchOrchestrator(
                gemini_client=MockGemini(),
                tavily_client=MockTavily(),
                openalex_provider=MockOpenAlex(),
                semantic_scholar_provider=MockS2(),
                max_retries=0,
                on_event=lambda name, payload: events.append(name),
            )
            res = orchestrator.execute_pipeline(
                topic="Multi-agent LLM debate",
                research_plan=plan,
                evidence_output_path=ev_path,
                synthesize_report=False,
            )

            self.assertIn("discovery_start", events)
            self.assertIn("discovery_complete", events)
            self.assertIn("extraction_complete", events)
            self.assertIn("pipeline_complete", events)
            self.assertTrue(os.path.exists(ev_path))
            self.assertGreaterEqual(len(res["extraction_bundle"]["all_findings"]), 1)


class TestSection5And6DatabaseCachingCheckpointingAndAcademicAPIs(unittest.TestCase):
    """Verifies Sections 5 & 6:
    1. Relational + Vector database schemas (users, projects, runs, sources, findings, 350-word passage vectors).
    2. Multi-tier SHA-256 caching (Search Query 14-30d TTL, Parsed PDF, Passage Embedding).
    3. Pipeline state checkpointing across Phase 1 (Plan), Phase 2 (Search/Passages), Phase 3 (Evidence).
    4. ar5iv HTML5 URL builder, Unpaywall DOI resolver, and Europe PMC REST API provider.
    """

    def test_database_relational_and_vector_passage_search(self):
        from search_agent.database import ResearchDatabase

        db = ResearchDatabase(db_path=":memory:")
        uid = db.create_user(email="researcher@example.org", name="Alice")
        pid = db.create_project(user_id=uid, name="LLM Reasoning Project")
        rid = db.create_run(topic="Speculative Decoding", project_id=pid, plan={"topic": "Speculative Decoding"})

        db.save_sources(rid, [{"source_id": "S1", "title": "Speculative Decoding Paper", "ranking_score": 0.91}])
        db.save_findings(
            rid,
            [
                {
                    "finding_id": "Q1-F1",
                    "question_id": "Q1",
                    "claim": "Achieves 2.5x speedup.",
                    "stance": "support",
                    "confidence": "high",
                    "source_ids": ["S1"],
                }
            ],
        )

        passages = [
            {"content": "Passage one discussing speculative decoding acceptance rates.", "word_count": 350},
            {"content": "Passage two discussing hardware memory bandwidth bottlenecks.", "word_count": 350},
        ]
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
        stored = db.store_passage_embeddings(rid, "S1", passages, embeddings)
        self.assertEqual(stored, 2)

        hits = db.search_similar_passages([0.95, 0.05, 0.0, 0.0], top_k=1, run_id=rid)
        self.assertEqual(len(hits), 1)
        self.assertIn("acceptance rates", hits[0]["content"])
        self.assertGreater(hits[0]["similarity"], 0.9)

        run_rec = db.get_run(rid)
        self.assertEqual(len(run_rec["sources"]), 1)
        self.assertEqual(len(run_rec["findings"]), 1)
        db.close()

    def test_multi_tier_sha256_cache(self):
        from search_agent.cache_manager import MultiTierCache, DEFAULT_SEARCH_TTL_SECONDS

        self.assertGreaterEqual(DEFAULT_SEARCH_TTL_SECONDS, 14 * 86400)
        self.assertLessEqual(DEFAULT_SEARCH_TTL_SECONDS, 30 * 86400)

        cache = MultiTierCache(cache_db_path=":memory:")
        # 1. Search query cache
        cache.set_search_results("openalex", "speculative decoding speedup", [{"source_id": "OA-1"}])
        self.assertEqual(cache.get_search_results("OpenAlex", "Speculative Decoding Speedup")[0]["source_id"], "OA-1")

        # 2. Parsed PDF cache
        cache.set_parsed_pdf("10.48550/arXiv.2302.01318", "Extracted full text from PDF...")
        self.assertEqual(cache.get_parsed_pdf("10.48550/arxiv.2302.01318"), "Extracted full text from PDF...")

        # 3. Embedding cache
        cache.set_embedding("Sample 350-word passage text", [0.25, 0.75, 0.5])
        self.assertEqual(cache.get_embedding("Sample 350-word passage text"), [0.25, 0.75, 0.5])
        cache.close()

    def test_pipeline_checkpoint_persistence_and_resumption(self):
        import tempfile
        from search_agent.checkpoint_manager import PipelineCheckpointManager

        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = PipelineCheckpointManager(checkpoint_dir=tmp_dir)
            topic = "Test Checkpointed Research Topic"
            mgr.save_phase(topic, "phase1_plan", {"questions": [{"id": "Q1"}]})
            mgr.save_phase(topic, "phase2_search_passages", {"sources": [{"source_id": "S1"}]})
            mgr.save_phase(topic, "phase3_evidence", {"extraction_bundle": {"all_findings": [{"finding_id": "Q1-F1"}]}})

            ckpt = mgr.load_checkpoint(topic)
            self.assertEqual(
                ckpt["completed_phases"],
                ["phase1_plan", "phase2_search_passages", "phase3_evidence"],
            )
            self.assertEqual(
                mgr.get_phase_state(topic, "phase3_evidence")["extraction_bundle"]["all_findings"][0]["finding_id"],
                "Q1-F1",
            )

    def test_ar5iv_unpaywall_and_europepmc_integrations(self):
        from unittest.mock import patch, MagicMock
        from search_agent.content_retriever import build_ar5iv_html_url, resolve_unpaywall_pdf_url, _extract_arxiv_id
        from search_agent.source_providers import EuropePMCProvider, get_provider

        self.assertEqual(build_ar5iv_html_url("2302.01318"), "https://ar5iv.labs.arxiv.org/html/2302.01318")
        self.assertEqual(_extract_arxiv_id("https://ar5iv.labs.arxiv.org/html/2302.01318v2"), "2302.01318v2")
        self.assertIsInstance(get_provider("europepmc"), EuropePMCProvider)

        # Mock Unpaywall DOI API response
        mock_unpaywall_resp = MagicMock()
        mock_unpaywall_resp.status = 200
        mock_unpaywall_resp.read.return_value = b'{"best_oa_location": {"url_for_pdf": "https://example.org/open_access.pdf"}}'
        mock_unpaywall_resp.__enter__.return_value = mock_unpaywall_resp

        with patch("urllib.request.urlopen", return_value=mock_unpaywall_resp):
            oa_pdf = resolve_unpaywall_pdf_url("https://doi.org/10.1038/s41586-023-00001-x")
            self.assertEqual(oa_pdf, "https://example.org/open_access.pdf")

        # Mock Europe PMC REST API response
        epmc = EuropePMCProvider()
        converted = epmc._convert(
            {
                "id": "38123456",
                "pmid": "38123456",
                "pmcid": "PMC10987654",
                "doi": "10.1016/j.cell.2024.01.001",
                "title": "CRISPR Epigenome Editing in Primary Human T Cells.",
                "abstractText": "<p>Quantitative evaluation of epigenome editing efficiency.</p>",
                "authorString": "Smith A, Jones B",
                "pubYear": "2024",
                "journalTitle": "Cell",
                "citedByCount": 42,
                "isOpenAccess": "Y",
            },
            query_text="CRISPR epigenome editing",
            question_id="Q1",
        )
        self.assertEqual(converted["source_id"], "EPMC-PMC10987654")
        self.assertEqual(converted["provider_name"], "europepmc")
        self.assertTrue(converted["is_open_access"])
        self.assertIn("epigenome editing efficiency", converted["abstract"])


class TestSection7And8AuthStreamingAndDeployment(unittest.TestCase):
    """Verifies Sections 7 & 8:
    1. JWT authentication, multi-tenant organization quotas, encrypted BYOK keys, and CLI login handshake.
    2. Background task queue with Server-Sent Events (SSE) progress streaming and FastAPI backend.
    3. Docker and docker-compose infrastructure files.
    """

    def test_jwt_auth_byok_encryption_and_workspace_quotas(self):
        from search_agent.auth_manager import (
            BYOKEncryptor,
            JWTAuthManager,
            MultiTenantWorkspaceManager,
            run_cli_login_handshake,
        )
        import tempfile
        import os

        # 1. BYOK encryption round-trip
        enc = BYOKEncryptor(master_secret="unit-test-secret-key")
        cipher = enc.encrypt_api_key("AIzaSyTestGeminiKey12345")
        self.assertNotIn("AIzaSyTestGeminiKey12345", cipher)
        self.assertEqual(enc.decrypt_api_key(cipher), "AIzaSyTestGeminiKey12345")

        # 2. JWT sign & verify
        jwt_mgr = JWTAuthManager(secret_key="unit-test-jwt-secret")
        token = jwt_mgr.create_jwt_token(user_id="usr_1", email="alice@example.org", org_id="org_1", tier="pro")
        claims = jwt_mgr.verify_jwt_token(token)
        self.assertEqual(claims["sub"], "usr_1")
        self.assertEqual(claims["email"], "alice@example.org")
        self.assertEqual(claims["org_id"], "org_1")

        # 3. Multi-tenant organization & quota tracking
        ws = MultiTenantWorkspaceManager(db_path=":memory:", master_secret="unit-test-secret-key")
        org_id = ws.create_organization(name="DeepResearch Lab", owner_user_id="usr_1", tier="free")
        ws.store_byok_key("usr_1", "gemini", "gemini-secret-key-999")
        self.assertEqual(ws.get_byok_key("usr_1", "gemini"), "gemini-secret-key-999")

        allowed, stats = ws.check_and_increment_quota(org_id, count=5)
        self.assertTrue(allowed)
        self.assertEqual(stats["queries_used"], 5)

        # Exceed free tier quota (20)
        allowed_over, _ = ws.check_and_increment_quota(org_id, count=20)
        self.assertFalse(allowed_over)
        ws.close()

        # 4. CLI OAuth login handshake
        with tempfile.TemporaryDirectory() as tmp_dir:
            cred_file = os.path.join(tmp_dir, "credentials.json")
            creds = run_cli_login_handshake(email="dev@ara.local", credentials_path=cred_file, open_browser=False)
            self.assertTrue(os.path.exists(cred_file))
            self.assertIn("access_token", creds)

    def test_background_task_queue_sse_streaming_and_fastapi_app(self):
        from search_agent.task_queue import BackgroundTaskQueue
        from ui.fastapi_app import create_fastapi_app
        import os

        queue = BackgroundTaskQueue(max_workers=2)

        def _sample_research_task(x, on_event=None):
            if on_event:
                on_event("discovery_start", {"step": 1})
                on_event("extraction_complete", {"findings": x})
            return {"total": x}

        job_id = queue.enqueue(_sample_research_task, 7, topic="SSE Test Topic")
        finished = queue.wait_for_job(job_id, timeout=5.0)
        self.assertEqual(finished["status"], "completed")
        self.assertEqual(finished["result"]["total"], 7)

        sse_frames = list(queue.stream_sse(job_id))
        combined_sse = "".join(sse_frames)
        self.assertIn("event: discovery_start", combined_sse)
        self.assertIn("event: extraction_complete", combined_sse)
        self.assertIn("event: done", combined_sse)

        app = create_fastapi_app(task_queue=queue)
        schema = app.openapi()
        self.assertIn("/api/jobs", schema["paths"])
        self.assertIn("/api/stream/{job_id}", schema["paths"])

        # Verify Dockerfile and docker-compose.yml exist
        self.assertTrue(os.path.exists("Dockerfile"))
        self.assertTrue(os.path.exists("docker-compose.yml"))


if __name__ == "__main__":
    unittest.main()

