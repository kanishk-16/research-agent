"""
demo/demo_data.py

Deterministic mock data for ARA offline demo mode (--demo).
Represents a realistic RAG hallucination vs factual degradation research session.
All citations, findings, quantitative records, and contradictions are structured
for offline testing and demonstration without external API dependencies.
"""

DEMO_TOPIC = "Under what conditions does retrieval-augmented generation reduce hallucination in large language models, and when can retrieval actually degrade factual accuracy?"

DEMO_PLAN = {
    "topic": DEMO_TOPIC,
    "domain": "AI/ML",
    "research_mode": "EVALUATIVE",
    "temporal_sensitivity": "MEDIUM",
    "hypothesis": "Retrieval-augmented generation (RAG) significantly reduces hallucination when retrieved contexts are highly relevant and accurate; however, retrieval degrades factual accuracy when retrieved passages contain distractors, noise, or outdated facts, leading models to override parametric memory.",
    "competing_hypotheses": [
        "Retrieval always improves factual accuracy across model architectures regardless of context noise.",
        "Model scale and alignment training completely immunize modern LLMs against adversarial retrieval distractors."
    ],
    "falsification_criteria": [
        "Empirical benchmarks showing zero factual degradation when injecting contradictory distractor passages.",
        "Systematic trials where irrelevant context injection never induces hallucination across tested model scales."
    ],
    "global_source_requirements": {
        "minimum_total_sources": 8,
        "minimum_primary_sources": 5,
        "minimum_peer_reviewed_sources": 6,
        "minimum_quantitative_sources": 5,
        "minimum_counter_evidence_sources": 2
    },
    "source_diversity_requirements": [
        "benchmark evaluations",
        "model scale variations",
        "retrieval noise types"
    ],
    "questions": [
        {
            "id": "Q1",
            "type": "background",
            "priority": "high",
            "question": "What standard benchmarks and evaluation metrics are used to measure hallucination and factual accuracy in RAG systems?",
            "evidence_needed": ["benchmark datasets", "evaluation frameworks", "accuracy metrics"],
            "source_requirements": {
                "minimum_sources": 3,
                "minimum_primary_sources": 2,
                "minimum_quantitative_sources": 1,
                "minimum_counter_evidence_sources": 0
            },
            "search_strategy": {
                "primary_queries": [
                    "RAG hallucination evaluation benchmarks",
                    "measuring factual accuracy large language models"
                ],
                "secondary_queries": [
                    "NaturalQuestions TriviaQA attribution faithfulness"
                ],
                "counter_evidence_queries": []
            },
            "requires_quantitative_evidence": False,
            "requires_counter_evidence": False
        },
        {
            "id": "Q2",
            "type": "quantitative",
            "priority": "high",
            "question": "What is the measured magnitude of factual accuracy reduction when large language models are presented with noisy, irrelevant, or distractor retrieved contexts?",
            "evidence_needed": ["effect sizes", "accuracy degradation metrics", "context noise experiments"],
            "source_requirements": {
                "minimum_sources": 4,
                "minimum_primary_sources": 3,
                "minimum_quantitative_sources": 3,
                "minimum_counter_evidence_sources": 1
            },
            "search_strategy": {
                "primary_queries": [
                    "retrieval noise distractor context factual degradation",
                    "LLM accuracy reduction irrelevant retrieved context"
                ],
                "secondary_queries": [
                    "lost in the middle context position recall degradation"
                ],
                "counter_evidence_queries": [
                    "frontier model scale robustness to retrieval noise",
                    "absence of factual degradation under distractor context"
                ]
            },
            "requires_quantitative_evidence": True,
            "quantitative_fields": ["metric", "baseline", "effect_size"],
            "requires_counter_evidence": True
        },
        {
            "id": "Q3",
            "type": "limitations",
            "priority": "medium",
            "question": "Under what boundary conditions, model scales, or architectural choices do LLMs successfully reject contradictory or misleading retrieved documents?",
            "evidence_needed": ["model scale ablation", "parametric vs non-parametric conflicts", "context filtering"],
            "source_requirements": {
                "minimum_sources": 3,
                "minimum_primary_sources": 2,
                "minimum_quantitative_sources": 1,
                "minimum_counter_evidence_sources": 1
            },
            "search_strategy": {
                "primary_queries": [
                    "robust context filtering RAG hallucination mitigation",
                    "post-retrieval reranking factual grounding"
                ],
                "secondary_queries": [
                    "parametric non-parametric knowledge conflict LLM"
                ],
                "counter_evidence_queries": [
                    "failure of context filtering in production RAG",
                    "latency overhead context verification LLMs"
                ]
            },
            "requires_quantitative_evidence": True,
            "quantitative_fields": ["metric", "faithfulness_gain"],
            "requires_counter_evidence": True
        }
    ]
}

DEMO_ROUND1_SUFFICIENCY = [
    {
        "question_id": "Q1",
        "sufficient": True,
        "finding_count": 6,
        "quantitative_count": 2,
        "counter_count": 0,
        "missing_requirements": [],
        "evidence_gaps": []
    },
    {
        "question_id": "Q2",
        "sufficient": False,
        "finding_count": 2,
        "quantitative_count": 0,
        "counter_count": 1,
        "missing_requirements": ["quantitative_evidence"],
        "evidence_gaps": ["Only 0 independent sources with quantitative evidence, but 3 required."]
    },
    {
        "question_id": "Q3",
        "sufficient": False,
        "finding_count": 1,
        "quantitative_count": 0,
        "counter_count": 0,
        "missing_requirements": ["counter_evidence"],
        "evidence_gaps": ["No direct counter-evidence extracted investigating model scale robustness."]
    }
]

DEMO_ROUND2_SUFFICIENCY = [
    {
        "question_id": "Q1",
        "sufficient": True,
        "finding_count": 6,
        "quantitative_count": 2,
        "counter_count": 0,
        "missing_requirements": [],
        "evidence_gaps": []
    },
    {
        "question_id": "Q2",
        "sufficient": True,
        "finding_count": 5,
        "quantitative_count": 4,
        "counter_count": 1,
        "missing_requirements": [],
        "evidence_gaps": []
    },
    {
        "question_id": "Q3",
        "sufficient": True,
        "finding_count": 4,
        "quantitative_count": 2,
        "counter_count": 2,
        "missing_requirements": [],
        "evidence_gaps": []
    }
]

DEMO_SOURCES = [
    {
        "source_id": "S1",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni"],
        "publication_year": 2020,
        "venue": "Advances in Neural Information Processing Systems (NeurIPS 2020)",
        "source_type": "peer_reviewed_paper",
        "provider_name": "semanticscholar",
        "doi": "https://doi.org/10.48550/arXiv.2005.11401",
        "url": "https://arxiv.org/abs/2005.11401"
    },
    {
        "source_id": "S2",
        "title": "A Survey on Retrieval-Augmented Generation for Large Language Models: Architecture, Taxonomy, and Benchmarking",
        "authors": ["Yunfan Gao", "Yun Xiong", "Xinyu Gao", "Kewei Jia"],
        "publication_year": 2023,
        "venue": "IEEE Transactions on Knowledge and Data Engineering",
        "source_type": "systematic_review",
        "provider_name": "openalex",
        "doi": "https://doi.org/10.1109/TKDE.2024.3364928",
        "url": "https://arxiv.org/abs/2312.10997"
    },
    {
        "source_id": "S3",
        "title": "Large Language Models Easily Distracted by Irrelevant Context",
        "authors": ["Freda Shi", "Xinyun Chen", "Kanishka Misra", "Nathan Scales"],
        "publication_year": 2023,
        "venue": "International Conference on Machine Learning (ICML 2023)",
        "source_type": "peer_reviewed_paper",
        "provider_name": "semanticscholar",
        "doi": "https://doi.org/10.48550/arXiv.2302.00093",
        "url": "https://proceedings.mlr.press/v202/shi23a.html"
    },
    {
        "source_id": "S4",
        "title": "Making Large Language Models Better Reasoners with Robust Context Filtering",
        "authors": ["Ori Yoran", "Tomer Wolfson", "Ori Ram", "Jonathan Berant"],
        "publication_year": 2024,
        "venue": "International Conference on Learning Representations (ICLR 2024)",
        "source_type": "peer_reviewed_paper",
        "provider_name": "semanticscholar",
        "doi": "https://doi.org/10.48550/arXiv.2305.14739",
        "url": "https://openreview.net/forum?id=7h823kLd"
    },
    {
        "source_id": "S5",
        "title": "Seven Failure Points in Retrieval-Augmented Generation Systems",
        "authors": ["Scott Barnett", "Stefanus Kurniawan", "Srikanth Thudumu"],
        "publication_year": 2024,
        "venue": "IEEE Software",
        "source_type": "peer_reviewed_paper",
        "provider_name": "openalex",
        "doi": "https://doi.org/10.1109/MS.2024.3382941",
        "url": "https://ieeexplore.ieee.org/document/10492817"
    },
    {
        "source_id": "S6",
        "title": "Lost in the Middle: How Language Models Use Long Contexts",
        "authors": ["Nelson F. Liu", "Kevin Lin", "John Hewitt", "Ashwin Paranjape"],
        "publication_year": 2024,
        "venue": "Transactions of the Association for Computational Linguistics (TACL)",
        "source_type": "peer_reviewed_paper",
        "provider_name": "semanticscholar",
        "doi": "https://doi.org/10.1162/tacl_a_00638",
        "url": "https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00638"
    },
    {
        "source_id": "S7",
        "title": "Frontier Language Models are Resilient to Contextual Misinformation Under Structured Prompting",
        "authors": ["Sophia Zhang", "Alexander Chen", "David Miller"],
        "publication_year": 2024,
        "venue": "Empirical Methods in Natural Language Processing (EMNLP 2024)",
        "source_type": "peer_reviewed_paper",
        "provider_name": "semanticscholar",
        "doi": "https://doi.org/10.18653/v1/2024.emnlp-main.812",
        "url": "https://aclanthology.org/2024.emnlp-main.812"
    },
    {
        "source_id": "S8",
        "title": "Benchmark Suite for Evaluating Factual Faithfulness in RAG Architectures",
        "authors": ["Marcus Vance", "Elena Rostova"],
        "publication_year": 2024,
        "venue": "ACM Computing Surveys",
        "source_type": "academic",
        "provider_name": "tavily",
        "url": "https://dl.acm.org/doi/10.1145/3672911"
    }
]

DEMO_EVIDENCE_ARTIFACT = {
    "schema_version": "1.1",
    "topic": DEMO_TOPIC,
    "phase": "phase_2_researcher",
    "summary": "Experimental evidence establishes that retrieval-augmented generation significantly lowers hallucination rates when retrieved contexts possess high precision and semantic relevance [S1, S2]. However, retrieval causes quantifiable factual accuracy degradation when top-ranked passages contain distractor sentences, outdated facts, or contradictory assertions [S3, S5, S6]. Specifically, distractor contexts induce accuracy drops ranging between 12% and 34% across standard question-answering benchmarks, with middle-positioned passages suffering from attention degradation [S3, S6]. Conversely, counter-evidence indicates that frontier model scales combined with strict chain-of-thought verification exhibit resilience to moderate retrieval noise [S7].",
    "evidence_gaps": [
        "Granular interaction between context compression ratios and domain-specific knowledge conflicts requires further cross-architecture validation."
    ],
    "sufficiency_evaluation": DEMO_ROUND2_SUFFICIENCY,
    "contradictions": [
        {
            "question_id": "Q2",
            "description": "Empirical tension between model scale resilience and retrieval noise susceptibility under distractor injection.",
            "support_claims": [
                "Adding irrelevant or distractor context passages reduces question answering accuracy by up to 22.4% [S3].",
                "Intermediate distractor positions trigger significant attention degradation (Lost-in-the-Middle effect) [S6]."
            ],
            "counter_claims": [
                "Frontier scale reasoning models remain robust to distractor contexts, degrading less than 2.5% when guided by grounding prompts [S7]."
            ],
            "dimensions": {
                "support_datasets": ["NaturalQuestions", "HotpotQA"],
                "counter_datasets": ["FrontierBench 2024"],
                "support_models": ["LLaMA-2 13B", "GPT-3.5-Turbo"],
                "counter_models": ["GPT-4", "Claude 3.5 Sonnet"]
            }
        }
    ],
    "evidence_diversity": {
        "diversity_score": 0.82
    },
    "statistics": {
        "candidate_sources": 544,
        "canonical_sources": 392,
        "selected_sources": 13,
        "useful_sources": 8,
        "idle_sources": 5,
        "quantitative_findings": 6,
        "counter_findings": 3,
        "total_findings": 15,
        "evidence_diversity_score": 0.82,
        "epistemic_triplet": {
            "finding_count": 15,
            "source_count": 8,
            "independent_source_count": 7
        }
    },
    "sources": DEMO_SOURCES,
    "questions": [
        {
            "question_id": "Q1",
            "question": "What standard benchmarks and evaluation metrics are used to measure hallucination and factual accuracy in RAG systems?",
            "findings": [
                {
                    "finding_id": "Q1-F1",
                    "claim": "RAG benchmark frameworks standardly evaluate factual hallucination using datasets such as NaturalQuestions, TriviaQA, and HotpotQA alongside attribution faithfulness metrics [S1, S2].",
                    "stance": "context",
                    "confidence": "high",
                    "source_ids": ["S1", "S2"],
                    "quantitative_evidence": []
                },
                {
                    "finding_id": "Q1-F2",
                    "claim": "Modern hallucination evaluation separates parametric memory recall from non-parametric retrieval grounding [S2, S5].",
                    "stance": "support",
                    "confidence": "high",
                    "source_ids": ["S2", "S5"],
                    "quantitative_evidence": []
                }
            ]
        },
        {
            "question_id": "Q2",
            "question": "What is the measured magnitude of factual accuracy reduction when large language models are presented with noisy, irrelevant, or distractor retrieved contexts?",
            "findings": [
                {
                    "finding_id": "Q2-F1",
                    "claim": "Injecting irrelevant distractor contexts into the prompt causes a statistically significant degradation in question answering accuracy of up to 22.4% (p < 0.001) across multiple benchmark datasets [S3].",
                    "stance": "support",
                    "confidence": "high",
                    "source_ids": ["S3"],
                    "quantitative_evidence": [
                        {
                            "metric": "QA Exact Match Accuracy",
                            "value": "51.8%",
                            "baseline": "74.2% (Oracle Context)",
                            "effect_size": "-22.4% Degradation (p < 0.001)",
                            "context": "Multi-hop QA benchmarks with 5 distractor passages"
                        }
                    ]
                },
                {
                    "finding_id": "Q2-F2",
                    "claim": "Position of retrieved information within long context windows substantially moderates performance: information placed in the middle yields degradation of 15% to 30% compared to beginning or end placements [S6].",
                    "stance": "support",
                    "confidence": "high",
                    "source_ids": ["S6"],
                    "quantitative_evidence": [
                        {
                            "metric": "Context Position Recall",
                            "value": "54.1%",
                            "baseline": "82.6% (Edge Position)",
                            "effect_size": "-28.5% Recall Deficit",
                            "context": "20-document context windows on NaturalQuestions"
                        }
                    ]
                },
                {
                    "finding_id": "Q2-F3",
                    "claim": "Frontier-scale aligned models maintain factual accuracy within 2.5% of baseline when exposed to distractor noise, demonstrating that model scale acts as a buffer against retrieval poisoning [S7].",
                    "stance": "counter",
                    "confidence": "high",
                    "source_ids": ["S7"],
                    "quantitative_evidence": [
                        {
                            "metric": "Distractor Robustness Delta",
                            "value": "89.2%",
                            "baseline": "91.7% (Clean Retrieval)",
                            "effect_size": "-2.5% Margin (Robust)",
                            "context": "Frontier reasoning model evaluation under adversarial distractors"
                        }
                    ]
                }
            ]
        },
        {
            "question_id": "Q3",
            "question": "Under what boundary conditions, model scales, or architectural choices do LLMs successfully reject contradictory or misleading retrieved documents?",
            "findings": [
                {
                    "finding_id": "Q3-F1",
                    "claim": "Context filtering and post-retrieval reranking algorithms improve factual grounding accuracy by 18.6%, neutralizing distractor interference [S4].",
                    "stance": "support",
                    "confidence": "high",
                    "source_ids": ["S4"],
                    "quantitative_evidence": [
                        {
                            "metric": "Faithfulness Gain",
                            "value": "+18.6%",
                            "baseline": "Unfiltered RAG",
                            "effect_size": "Cohen's d = 0.68",
                            "context": "ICLR robust context filtering benchmark"
                        }
                    ]
                },
                {
                    "finding_id": "Q3-F2",
                    "claim": "Architectural failure points in production RAG systems stem predominantly from passage chunk boundary truncation rather than embedding space misalignment [S5].",
                    "stance": "context",
                    "confidence": "moderate",
                    "source_ids": ["S5"],
                    "quantitative_evidence": []
                }
            ]
        }
    ]
}

DEMO_REPORT_ARTIFACT = {
    "schema_version": "1.0",
    "phase": "phase_3_synthesizer",
    "topic": DEMO_TOPIC,
    "epistemic_calibration": {
        "confidence_score": 0.78,
        "confidence_tier": "MODERATE",
        "tier_description": "Moderate Epistemic Confidence: High consistency across benchmark studies, with clearly defined boundary conditions regarding context distractors and model scale.",
        "sufficiency_ratio": 1.0,
        "independence_ratio": 0.88,
        "diversity_score": 0.82,
        "contradictions_detected": 1
    },
    "report_markdown": f"""# {DEMO_TOPIC}

## 1. Executive Summary & Epistemic Calibration
This research investigation evaluates the conditions under which retrieval-augmented generation (RAG) mitigates factual hallucination in large language models (LLMs), alongside identifying the critical thresholds where retrieval degrades generation accuracy. Operating under a **MODERATE** epistemic calibration level (Confidence Score: 0.78 / 1.0), empirical findings confirm that RAG universally improves factual precision when retrieved passages are highly relevant and structurally concise [S1, S2]. However, significant factual degradation occurs when top-ranked context contains distractor passages, intermediate placement bias, or contradictory assertions [S3, S6]. 

---

## 2. Evidence Analysis by Sub-Question

### Q1: Standard Benchmarks and Metrics
Hallucination and factual grounding in RAG are empirically measured across standardized benchmarks including **NaturalQuestions**, **TriviaQA**, and **HotpotQA** [S1, S2]. Attribution metrics track exact match (EM), F1 faithfulness, and attribution coverage. Evaluation distinguishes between internal parametric memory access and context-grounded non-parametric generation [S2, S5].

### Q2: Magnitude of Retrieval-Induced Factual Degradation
The injection of irrelevant or contradictory distractor passages causes quantifiable accuracy drops. Empirical evaluations show exact match accuracy declining from 74.2% under oracle conditions down to 51.8% when five distractor passages are introduced (-22.4% degradation, p < 0.001) [S3]. Furthermore, document placement within the context window triggers a severe 'Lost-in-the-Middle' effect, where facts placed centrally suffer 15% to 30% lower recall compared to edge placements [S6].

### Q3: Boundary Conditions and Mitigation Strategies
Model scale and contextual filtering act as primary moderators of degradation. Post-retrieval context reranking and filtering eliminate noisy passages, yielding an 18.6% improvement in faithful generation over standard RAG pipelines [S4]. Furthermore, counter-evidence confirms that frontier-scale reasoning models exhibit intrinsic resilience against distractor interference, maintaining accuracy within 2.5% of clean retrieval baselines [S7].

---

## 3. Quantitative Evaluation & Comparative Contrasts

| Metric / Parameter | Clean Retrieval (Oracle) | Noisy / Distractor Retrieval | Filtered RAG Pipeline |
| :--- | :--- | :--- | :--- |
| **QA Exact Match** | 74.2% [S3] | 51.8% (-22.4% delta) [S3] | 70.4% (+18.6% gain) [S4] |
| **Position Sensitivity** | 82.6% (Edge) [S6] | 54.1% (Center) [S6] | Stabilized [S4] |
| **Frontier Model Robustness** | 91.7% [S7] | 89.2% (-2.5% delta) [S7] | 92.4% [S7] |

---

## 4. Counter-Evidence, Trade-Offs & Falsification Analysis
* **Model Scale Robustness Paradox:** While small and medium-scale models (7B to 13B) suffer severe performance degradation under distractor context injection [S3], frontier models demonstrate high resilience to adversarial distraction when paired with chain-of-thought verification [S7].
* **Latency vs. Accuracy Trade-Off:** Intensive post-retrieval context filtering and validation resolve distractor issues [S4], but introduce an approximate 2.4x latency penalty in production systems [S5].

---

## 5. Methodological Limitations & Source Independence
1. **Benchmark Artificiality:** Benchmark evaluations typically inject synthetically formatted distractors rather than natural semantic drifts encountered in live web corpora [S3, S8].
2. **Context Window Saturation:** Many findings were measured on models with 4k-8k context windows; behavior under 100k+ token windows requires independent replication [S6].

---

## 6. Epistemic Conclusion
Retrieval-augmented generation is an effective countermeasure against parametric hallucination [S1], but becomes an active liability when retriever precision falls below 65% in the presence of distractor noise [S3, S6]. Production RAG deployments must incorporate robust post-retrieval filtering [S4] or leverage frontier reasoning models [S7] to prevent retrieval-induced degradation.""",
    "evidence_reference": "data/research_evidence.json",
    "plan_reference": "data/research_plan.json"
}
