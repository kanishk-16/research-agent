"""
search_agent/orchestrator.py

Unified ResearchOrchestrator service class eliminating duplicated Phase 2 and Phase 3
pipeline execution logic between CLI (`main.py`) and Web UI (`ui/server.py`).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from .content_retriever import retrieve_selected_sources
from .evidence_extractor import MODEL as DEFAULT_EXTRACTOR_MODEL, extract_research_evidence
from .evidence_output import build_research_evidence_output, save_research_evidence
from .evidence_sufficiency import check_all_sufficiency, generate_targeted_queries
from .evidence_validator import EvidenceValidator
from .researcher import ResearchPackage, run_research, run_targeted_research
from .source_providers import OpenAlexProvider, SemanticScholarProvider
from .source_ranker import print_ranking_diagnostics, rank_sources
from .source_selector import select_sources
from .sources import deduplicate_sources
from .summary_generator import detect_empirical_contradictions, generate_research_summary


class ResearchOrchestrator:
    """
    Shared pipeline orchestrator service coordinating:
    1. Multi-provider discovery, deduplication, ranking, selection, and full-text retrieval
    2. Pydantic structured evidence extraction and EVL 5-gate validation
    3. Evidence sufficiency evaluation and gap-driven targeted re-search rounds
    4. Research summary, evidence artifact, calibrated synthesis, and HTML report generation
    """

    def __init__(
        self,
        gemini_client: Any,
        tavily_client: Any,
        openalex_provider: Optional[Any] = None,
        semantic_scholar_provider: Optional[Any] = None,
        model: str = DEFAULT_EXTRACTOR_MODEL,
        max_retries: int = 1,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.gemini_client = gemini_client
        self.tavily_client = tavily_client
        self.openalex_provider = openalex_provider if openalex_provider is not None else OpenAlexProvider()
        self.semantic_scholar_provider = (
            semantic_scholar_provider
            if semantic_scholar_provider is not None
            else SemanticScholarProvider(tavily_client=tavily_client)
        )
        self.model = model
        self.max_retries = max_retries
        self.on_event = on_event

    def _emit(self, event_name: str, **payload: Any) -> None:
        if self.on_event is not None:
            try:
                self.on_event(event_name, payload)
            except Exception:
                pass

    def execute_discovery(self, topic: str, research_plan: Dict[str, Any]) -> ResearchPackage:
        """Execute Phase 2 initial multi-provider search, ranking, selection, and content retrieval."""
        questions = research_plan.get("questions", [])
        self._emit("discovery_start", topic=topic, question_count=len(questions))
        research_pkg = run_research(
            self.tavily_client,
            topic,
            questions,
            self.gemini_client,
            research_plan=research_plan,
            openalex_provider=self.openalex_provider,
            semantic_scholar_provider=self.semantic_scholar_provider,
        )
        self._emit(
            "discovery_complete",
            research_pkg=research_pkg,
            source_count=len(research_pkg),
        )
        return research_pkg

    def execute_targeted_research_rounds(
        self,
        topic: str,
        research_plan: Dict[str, Any],
        research_pkg: ResearchPackage,
        selection_bundle: Dict[str, Any],
        extraction_bundle: Dict[str, Any],
        sufficiency_results: List[Dict[str, Any]],
        insufficient_questions: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        verbose_diagnostics: bool = False,
    ) -> Tuple[ResearchPackage, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Tuple[Dict[str, Any], Dict[str, Any]]]]:
        """Execute gap-driven targeted re-search rounds when initial evidence is insufficient."""
        retry_attempt = 0
        all_questions = research_plan.get("questions", [])
        source_policy = research_plan.get("source_policy") if isinstance(research_plan, dict) else None

        while insufficient_questions and retry_attempt < self.max_retries:
            retry_attempt += 1
            gap_entries = [
                {
                    "question_id": sr["question_id"],
                    "missing_requirements": sr["missing_requirements"],
                }
                for _, sr in insufficient_questions
            ]
            self._emit("evidence_gaps_detected", attempt=retry_attempt, gap_entries=gap_entries)

            targeted_entries = []
            query_actions = []
            for question, sr in insufficient_questions:
                missing_reqs = sr.get("missing_requirements", [])
                targeted_queries = generate_targeted_queries(question, missing_reqs)
                if not targeted_queries:
                    continue
                targeted_entries.append({
                    "question": question,
                    "targeted_queries": targeted_queries,
                })
                for tq in targeted_queries:
                    q_lower = tq.get("query_text", "").lower()
                    req_type = (
                        "quantitative"
                        if "quantitative" in q_lower
                        else ("counter" if "counter" in tq.get("query_type", "").lower() else "targeted")
                    )
                    query_actions.append(f"searching {req_type} evidence")
                self._emit(
                    "targeted_queries_generated",
                    question_id=sr.get("question_id", "?"),
                    queries=targeted_queries,
                    missing_requirements=missing_reqs,
                )

            if not targeted_entries:
                break

            self._emit("targeted_research_start", attempt=retry_attempt, query_actions=query_actions)
            targeted_results = run_targeted_research(
                self.tavily_client,
                topic,
                targeted_entries,
                gemini_client=self.gemini_client,
                research_plan=research_plan,
                openalex_provider=self.openalex_provider,
                semantic_scholar_provider=self.semantic_scholar_provider,
            )
            if not targeted_results:
                break

            existing_sources = list(
                selection_bundle.get("canonical_sources", list(research_pkg))
            )
            merged_deduped = deduplicate_sources(existing_sources + list(targeted_results))
            merged_ranked = rank_sources(merged_deduped, all_questions, source_policy=source_policy)
            merged_selection = select_sources(all_questions, merged_ranked)

            if verbose_diagnostics:
                print_ranking_diagnostics(merged_ranked, all_questions, top_n=5)

            merged_retrieval = retrieve_selected_sources(
                self.tavily_client,
                merged_selection["unique_selected_sources"],
            )
            self._emit(
                "targeted_retrieval_complete",
                retrieval_stats=merged_retrieval,
                selected_count=len(merged_selection["unique_selected_sources"]),
                merged_count=len(merged_deduped),
            )

            extraction_bundle = extract_research_evidence(
                self.gemini_client,
                research_plan,
                merged_selection,
                model=self.model,
                existing_extraction_bundle=extraction_bundle,
            )
            sufficiency_results, insufficient_questions = check_all_sufficiency(
                all_questions,
                extraction_bundle,
                merged_selection,
            )
            self._emit(
                "targeted_sufficiency_evaluated",
                sufficiency_results=sufficiency_results,
                insufficient_questions=insufficient_questions,
            )

            research_pkg = ResearchPackage(
                merged_ranked,
                selection_bundle=merged_selection,
                unique_selected_sources=merged_selection["unique_selected_sources"],
                retrieval_stats=merged_retrieval,
            )
            selection_bundle = merged_selection

        return (
            research_pkg,
            selection_bundle,
            extraction_bundle,
            sufficiency_results,
            insufficient_questions,
        )

    def execute_pipeline(
        self,
        topic: str,
        research_plan: Dict[str, Any],
        evidence_output_path: Optional[str] = "data/research_evidence.json",
        synthesize_report: bool = True,
        html_report_path: Optional[str] = "data/research_report.html",
        verbose_diagnostics: bool = False,
        checkpoint_manager: Optional[Any] = None,
        run_id: Optional[str] = None,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the full Phase 2 & Phase 3 research pipeline and return all resulting artifacts.
        Supports optional PipelineCheckpointManager state persistence and crash resumption.
        """
        questions = research_plan.get("questions", [])
        if checkpoint_manager is not None:
            checkpoint_manager.save_phase(topic, "phase1_plan", {"research_plan": research_plan}, run_id=run_id)

        restored_phase2 = (
            checkpoint_manager.get_phase_state(topic, "phase2_search_passages", run_id=run_id)
            if (checkpoint_manager is not None and resume)
            else None
        )
        if restored_phase2 and restored_phase2.get("sources"):
            selection_bundle = restored_phase2["selection_bundle"]
            research_pkg = ResearchPackage(
                restored_phase2["sources"],
                selection_bundle=selection_bundle,
                unique_selected_sources=selection_bundle.get("unique_selected_sources", []),
                retrieval_stats=restored_phase2.get("retrieval_stats", {}),
            )
        else:
            research_pkg = self.execute_discovery(topic, research_plan)
            if verbose_diagnostics:
                print_ranking_diagnostics(research_pkg, questions)
            selection_bundle = getattr(research_pkg, "selection_bundle", {})
            if checkpoint_manager is not None:
                checkpoint_manager.save_phase(
                    topic,
                    "phase2_search_passages",
                    {
                        "sources": list(research_pkg),
                        "selection_bundle": selection_bundle,
                        "retrieval_stats": getattr(research_pkg, "retrieval_stats", {}),
                    },
                    run_id=run_id,
                )

        restored_phase3 = (
            checkpoint_manager.get_phase_state(topic, "phase3_evidence", run_id=run_id)
            if (checkpoint_manager is not None and resume)
            else None
        )
        if restored_phase3 and restored_phase3.get("extraction_bundle"):
            extraction_bundle = restored_phase3["extraction_bundle"]
            sufficiency_results = restored_phase3.get("sufficiency_results", [])
            insufficient_questions = []
        else:
            self._emit("extraction_start")
            extraction_bundle = extract_research_evidence(
                self.gemini_client,
                research_plan,
                selection_bundle,
                model=self.model,
            )
            self._emit(
                "extraction_complete",
                extraction_bundle=extraction_bundle,
                finding_count=len(extraction_bundle.get("all_findings", [])),
            )

            self._emit("sufficiency_start")
            sufficiency_results, insufficient_questions = check_all_sufficiency(
                questions,
                extraction_bundle,
                selection_bundle,
            )
            self._emit(
                "sufficiency_complete",
                sufficiency_results=sufficiency_results,
                insufficient_questions=insufficient_questions,
            )
            if checkpoint_manager is not None:
                checkpoint_manager.save_phase(
                    topic,
                    "phase3_evidence",
                    {
                        "extraction_bundle": extraction_bundle,
                        "sufficiency_results": sufficiency_results,
                    },
                    run_id=run_id,
                )

        if insufficient_questions and self.max_retries > 0:
            (
                research_pkg,
                selection_bundle,
                extraction_bundle,
                sufficiency_results,
                insufficient_questions,
            ) = self.execute_targeted_research_rounds(
                topic=topic,
                research_plan=research_plan,
                research_pkg=research_pkg,
                selection_bundle=selection_bundle,
                extraction_bundle=extraction_bundle,
                sufficiency_results=sufficiency_results,
                insufficient_questions=insufficient_questions,
                verbose_diagnostics=verbose_diagnostics,
            )

        self._emit("summary_start")
        selected_sources = getattr(research_pkg, "unique_selected_sources", list(research_pkg))
        useful_sources, _ = EvidenceValidator.filter_useful_sources(
            selected_sources,
            extraction_bundle.get("all_findings", []),
        )
        sources_for_summary = useful_sources if useful_sources else selected_sources
        contradictions = detect_empirical_contradictions(extraction_bundle.get("all_findings", []))

        try:
            summary = generate_research_summary(
                self.gemini_client,
                topic,
                sources_for_summary,
                extraction_bundle=extraction_bundle,
                sufficiency_results=sufficiency_results,
                contradictions=contradictions,
                model=self.model,
            )
            self._emit("summary_complete", summary=summary)
        except Exception as summary_exc:
            all_findings = extraction_bundle.get("all_findings", [])
            finding_bullets = "\n".join(
                f"- [{f.get('finding_id', 'F')}] ({f.get('stance', 'context').upper()}): {f.get('claim', '')}"
                for f in all_findings[:10]
            )
            summary = (
                f"[Synthesis temporarily unavailable — retry later. Extracted {len(all_findings)} findings across "
                f"{len(sources_for_summary)} sources for '{topic}'.]\n\n"
                f"**Extracted Findings Preview:**\n{finding_bullets or 'No structured findings extracted.'}"
            )
            self._emit("summary_warning", error=str(summary_exc), summary=summary)

        evidence_artifact = build_research_evidence_output(
            topic,
            research_pkg,
            extraction_bundle,
            summary=summary,
            sufficiency_results=sufficiency_results,
            research_plan=research_plan,
        )
        if evidence_output_path:
            save_research_evidence(evidence_artifact, evidence_output_path)
            try:
                from .database import ResearchDatabase
                db = ResearchDatabase()
                rid = db.create_run(topic=topic, status="completed", plan=research_plan, run_id=run_id)
                db.save_sources(rid, list(research_pkg))
                db.save_findings(rid, extraction_bundle.get("all_findings", []))
                db.close()
            except Exception:
                pass

        report_artifact = None
        if synthesize_report:
            from synthesizer.calibrated_synthesizer import CalibratedSynthesizer, save_research_report
            from reports.html_renderer import render_html_report

            synthesizer = CalibratedSynthesizer(
                gemini_client=self.gemini_client,
                model=self.model,
            )
            report_artifact = synthesizer.synthesize(
                topic,
                research_plan,
                evidence_artifact,
            )
            save_research_report(report_artifact)
            if html_report_path:
                render_html_report(
                    report_artifact=report_artifact,
                    evidence_artifact=evidence_artifact,
                    research_plan=research_plan,
                    output_path=html_report_path,
                    is_demo=False,
                )

        self._emit(
            "pipeline_complete",
            evidence_artifact=evidence_artifact,
            report_artifact=report_artifact,
        )

        return {
            "research_pkg": research_pkg,
            "selection_bundle": selection_bundle,
            "extraction_bundle": extraction_bundle,
            "sufficiency_results": sufficiency_results,
            "insufficient_questions": insufficient_questions,
            "summary": summary,
            "evidence_artifact": evidence_artifact,
            "report_artifact": report_artifact,
        }
