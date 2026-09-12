import re
from urllib.parse import urlsplit


SOURCE_TYPES = {
    "systematic_review",
    "peer_reviewed_paper",
    "conference_paper",
    "preprint",
    "government",
    "official",
    "academic",
    "standards",
    "documentation",
    "news",
    "technical_article",
    "corporate_blog",
    "personal_blog",
    "forum",
    "social_media",
    "video",
    "other"
}

SOURCE_QUALITY_SCORES = {
    "systematic_review": 1.00,
    "peer_reviewed_paper": 0.95,
    "conference_paper": 0.90,
    "government": 0.90,
    "official": 0.88,
    "standards": 0.88,
    "academic": 0.82,
    "preprint": 0.80,
    "documentation": 0.78,
    "technical_article": 0.65,
    "news": 0.55,
    "corporate_blog": 0.50,
    "personal_blog": 0.35,
    "forum": 0.25,
    "social_media": 0.20,
    "video": 0.20,
    "other": 0.40
}

RANKING_WEIGHTS = {
    "relevance": 0.40,
    "source_quality": 0.25,
    "planner_preference": 0.20,
    "tavily": 0.15
}

EVIDENCE_ELIGIBLE_SOURCE_TYPES = {
    "systematic_review",
    "peer_reviewed_paper",
    "conference_paper",
    "government",
    "official",
    "standards",
    "academic",
    "preprint",
    "documentation",
    "technical_article",
}

EVIDENCE_EXCLUSION_REASONS = {
    "video": "video",
    "social_media": "social_media",
    "forum": "forum",
    "personal_blog": "blog",
    "corporate_blog": "blog",
    "news": "news",
    "other": "unsupported_source_type",
}

PREFERENCE_TO_SOURCE_TYPES = {
    "systematic review": {"systematic_review"},
    "systematic survey": {"systematic_review"},
    "meta-analysis": {"systematic_review"},
    "peer-reviewed research paper": {"peer_reviewed_paper"},
    "peer-reviewed paper": {"peer_reviewed_paper"},
    "clinical study": {"peer_reviewed_paper"},
    "conference paper": {"conference_paper"},
    "benchmark paper": {"peer_reviewed_paper", "conference_paper"},
    "primary research preprint": {"preprint"},
    "preprint": {"preprint"},
    "official technical report": {"official", "government"},
    "technical report": {"official", "government", "academic"},
    "official documentation": {"documentation", "official"},
    "technical documentation": {"documentation", "official"},
    "technical standard": {"standards"},
    "security advisory": {"official", "government"},
    "government dataset": {"government"},
    "government report": {"government"},
    "academic research paper": {"peer_reviewed_paper", "academic"},
    "research paper": {"peer_reviewed_paper", "academic"}
}

STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "for", "from", "how",
    "in", "into", "is", "of", "on", "or", "that", "the", "this", "to",
    "under", "what", "when", "where", "which", "with"
}

_TERM_WEIGHTS = {
    "model": 0.1, "models": 0.1, "transformer": 0.1, "transformers": 0.1,
    "deep": 0.1, "learning": 0.1, "ai": 0.1, "artificial": 0.1,
    "intelligence": 0.1, "performance": 0.1, "approach": 0.1,
    "approaches": 0.1, "method": 0.1, "methods": 0.1, "based": 0.1,
    "using": 0.1, "new": 0.1, "paper": 0.1, "study": 0.1, "studies": 0.1,
    "result": 0.1, "results": 0.1, "system": 0.1, "systems": 0.1,
    "data": 0.1, "dataset": 0.1, "datasets": 0.1, "training": 0.1,
    "trained": 0.1, "accuracy": 0.1, "efficient": 0.1, "state": 0.1,
    "art": 0.1, "neural": 0.1, "network": 0.1, "networks": 0.1,
    "architecture": 0.1, "architectures": 0.1, "benchmark": 0.1,
    "benchmarks": 0.1, "task": 0.1, "tasks": 0.1, "machine": 0.1,
    "vision": 0.1, "image": 0.1, "images": 0.1, "video": 0.1,
    "videos": 0.1, "speech": 0.1, "recognition": 0.1, "generation": 0.1,
    "reinforcement": 0.1, "agent": 0.1, "agents": 0.1,
}

_NLP_QUERY_INDICATORS = {
    "nlp",
    "natural language",
    "language",
    "text",
    "translation",
    "linguistic",
    "sentiment",
    "parsing",
    "question",
    "bert",
    "gpt",
    "attention",
}

_NLP_DOMAIN_CONCEPTS = {
    "attention mechanism",
    "self-attention",
    "multi-head attention",
    "language model",
    "pretraining",
    "pre-training",
    "fine-tuning",
    "pretrained",
    "machine translation",
    "sequence-to-sequence",
    "seq2seq",
    "named entity",
    "question answering",
    "text generation",
    "language understanding",
    "natural language",
    "positional encoding",
    "wordpiece",
    "subword",
    "encoder-decoder",
    "recurrent",
    "convolutional",
    "sequence transduction",
    "bidirectional",
    "masked language",
    "language representation",
}

_NON_NLP_DOMAIN_INDICATORS = {
    "medical imaging",
    "medical image",
    "image segmentation",
    "computer vision",
    "pedestrian detection",
    "time series",
    "trajectory forecasting",
    "trajectory prediction",
    "asset prediction",
    "weather forecasting",
    "molecular biology",
    "remote sensing",
    "satellite imagery",
    "video understanding",
    "depth estimation",
    "point cloud",
    "graph neural",
    "electric vehicle",
    "ev charging",
    "load forecasting",
    "energy management",
    "autonomous driving",
    "robotics",
    "drug discovery",
    "smart grid",
    "power system",
}

_NON_NLP_SECONDARY_INDICATORS = {
    "speech",
    "audio",
    "music",
    "recommendation",
    "forecasting",
    "prediction",
}

_QUERY_INTENT_CONCEPTS = {
    "performance": {
        "benchmark", "evaluation", "performance", "accuracy", "f1",
        "bleu", "rouge", "glue", "superglue", "mmlu",
        "comparison", "baseline", "fine-tuning", "zero-shot", "few-shot",
        "language understanding", "natural language processing",
    },
    "computational_cost": {
        "inference", "training compute", "computational cost", "flops",
        "latency", "throughput", "memory", "gpu", "energy", "power",
        "efficiency", "carbon", "deployment", "serving", "quantization",
        "pruning", "distillation", "hardware", "inference cost",
    },
    "robustness": {
        "adversarial", "robustness", "attack", "vulnerability",
        "hallucination", "factuality", "out-of-distribution", "ood",
        "distribution shift", "generalization", "perturbation", "safety",
        "reliability",
    },
    "scaling": {
        "scaling law", "parameter scaling", "parameter count", "model size",
        "data scaling", "dataset size", "compute-optimal", "data availability",
        "architectural bottleneck", "parameter-efficient", "peft", "lora",
        "memory", "fine-tuning cost",
    },
    "hallucination_definition": {
        "hallucination", "factuality", "faithfulness", "factual consistency",
        "taxonomy", "classification", "detection", "evaluation",
        "hallucination types", "factual error", "unsupported claim",
        "contradiction", "grounding", "factuality error",
    },
    "rag_vs_nonrag_comparison": {
        "retrieval augmented generation", "rag", "baseline", "non-rag",
        "without retrieval", "with retrieval", "comparison", "ablation",
        "benchmark", "hallucination rate", "factuality score",
        "faithfulness score", "evaluation dataset", "empirical evaluation",
        "experimental results", "no retrieval", "vanilla",
        "retrieval-augmented", "non-retrieval",
    },
    "rag_failure_modes": {
        "retrieval failure", "irrelevant context", "noisy context",
        "outdated knowledge", "missing information", "incomplete context",
        "conflicting evidence", "context fragmentation", "chunking",
        "retrieval quality", "hallucination propagation",
        "retrieval-induced hallucination", "unsupported generation",
        "knowledge base limitation", "robustness", "failure mode",
        "boundary condition", "retrieval error", "context window",
    },
    "retrieval_configuration": {
        "dense retrieval", "sparse retrieval", "hybrid retrieval",
        "bm25", "vector search", "embedding retrieval", "top-k",
        "chunk size", "chunking strategy", "overlap", "reranking",
        "re-ranking", "cross-encoder", "retriever", "retrieval precision",
        "retrieval recall", "retrieval f1", "similarity threshold",
        "retrieval configuration", "dense", "sparse", "hybrid",
    },
}

_DOMAIN_MISMATCH_INDICATORS = {
    "protein language models",
    "genomics",
    "molecular",
    "medical imaging",
    "image classification",
    "computer vision",
    "traffic prediction",
    "remote sensing",
    "agriculture",
    "robotics",
    "speech emotion recognition",
    "drug discovery",
    "recommendation system",
    "sensory precision",
    "hallucination-prone",
    "perceptual",
    "psychology",
    "neuroscience",
    "sensory",
    "perception",
    "clinical",
    "psychiatric",
    "schizophrenia",
    "depression",
    "anxiety",
}


def _term_weight(term):
    return _TERM_WEIGHTS.get(term, 1.0)


def _is_nlp_question(question_text):
    qtext = str(question_text or "").lower()
    return any(indicator in qtext for indicator in _NLP_QUERY_INDICATORS)


def _is_llm_rag_question(question_text):
    qtext = str(question_text or "").lower()
    return any(
        indicator in qtext
        for indicator in (
            "llm", "large language model", "language model",
            "rag", "retrieval augmented", "retrieval",
            "hallucination", "factuality", "faithfulness",
        )
    )


def _detect_query_intent(question_text, question_type=None):
    qtype = str(question_type or "").lower().strip()
    qtext = str(question_text or "").lower()

    # RAG/hallucination-specific intent detection takes precedence over generic type
    if any(kw in qtext for kw in ("failure mode", "boundary condition", "fail to prevent", "retrieval configuration")):
        if "retrieval configuration" in qtext or "dense" in qtext or "sparse" in qtext or "chunking" in qtext:
            return "retrieval_configuration"
        return "rag_failure_modes"

    if any(kw in qtext for kw in ("rag vs", "non-rag", "without retrieval", "with retrieval", "compared to", "compared with")):
        return "rag_vs_nonrag_comparison"

    if any(kw in qtext for kw in ("how much", "to what extent", "how effectively", "how significantly")):
        if "rag" in qtext or "retrieval" in qtext:
            return "rag_vs_nonrag_comparison"

    if any(kw in qtext for kw in ("how do", "what retrieval", "what configuration", "dense vs", "sparse vs", "chunking", "reranking")):
        return "retrieval_configuration"

    if any(kw in qtext for kw in ("what are hallucinations", "how are hallucinations", "define", "definition of", "taxonomy")):
        return "hallucination_definition"

    if any(kw in qtext for kw in ("hallucination", "rag", "retrieval")):
        if "retrieval" in qtext and ("configuration" in qtext or "dense" in qtext or "sparse" in qtext or "chunk" in qtext):
            return "retrieval_configuration"
        if "failure" in qtext or "fail" in qtext or "boundary" in qtext:
            return "rag_failure_modes"
        if "compare" in qtext or "versus" in qtext or "vs" in qtext or "compared" in qtext:
            return "rag_vs_nonrag_comparison"
        if "what are" in qtext or "definition" in qtext or "taxonomy" in qtext or "how are" in qtext:
            return "hallucination_definition"
        return "robustness"

    if qtype == "performance":
        return "performance"
    elif qtype == "effects":
        if any(kw in qtext for kw in ("computational", "cost", "memory", "energy", "flops", "latency", "inference", "environmental")):
            return "computational_cost"
        return "performance"
    elif qtype == "limitations":
        return "robustness"
    elif qtype == "scaling":
        return "scaling"

    cost_keywords = ("computational", "cost", "memory", "energy", "flops", "latency", "inference", "environmental")
    perf_keywords = ("benchmark", "performance", "comparison", "accuracy", "evaluation")
    robust_keywords = ("robustness", "failure", "vulnerability", "hallucination", "generalization")
    scaling_keywords = ("scaling", "parameter", "model size", "compute-optimal")

    scores = {
        "computational_cost": sum(1 for kw in cost_keywords if kw in qtext),
        "performance": sum(1 for kw in perf_keywords if kw in qtext),
        "robustness": sum(1 for kw in robust_keywords if kw in qtext),
        "scaling": sum(1 for kw in scaling_keywords if kw in qtext),
    }

    best_intent = max(scores, key=scores.get)
    if scores[best_intent] > 0:
        return best_intent

    return None


def _domain_and_text(source):
    url = str(source.get("url", "") or "").strip()
    parsed = urlsplit(url)
    domain = (parsed.hostname or "").lower()
    text = " ".join(
        str(source.get(field, "") or "")
        for field in ("title", "content")
    ).lower()
    return domain, text


def classify_source(source):
    """Classify a source using conservative URL and metadata signals."""

    provider_name = str(source.get("provider_name", "") or "").lower()
    venue = str(source.get("venue", "") or "").lower()
    doi = str(source.get("doi", "") or "").lower()

    if provider_name == "openalex":
        if venue:
            conference_indicators = (
                "conference", "proceedings",
                "acl", "emnlp", "naacl", "aaai", "ijcai",
                "neurips", "nips", "icml", "iclr",
                "cvpr", "iccv", "eccv", "icassp",
                "colt", "kdd", "sigir", "www",
            )
            journal_indicators = (
                "journal", "transactions", "letters",
                "review", "annals", "bulletin",
            )

            if any(ind in venue for ind in conference_indicators):
                return "conference_paper"
            elif any(ind in venue for ind in journal_indicators):
                return "peer_reviewed_paper"

        return "academic"

    domain, text = _domain_and_text(source)
    title = str(source.get("title", "") or "").lower()
    combined = f"{title} {text}"

    if domain in {"arxiv.org", "export.arxiv.org"} and "/abs/" in str(source.get("url", "")):
        return "preprint"

    if "aclanthology.org" in domain or "proceedings" in domain or "openreview.net" in domain:
        return "conference_paper"

    if domain.endswith(".gov") or ".gov." in domain:
        return "government"

    if domain in {"reddit.com", "www.reddit.com", "old.reddit.com"} or domain.endswith(".reddit.com"):
        return "forum"

    if domain in {"linkedin.com", "www.linkedin.com"} or domain.endswith(".linkedin.com"):
        return "social_media"

    if domain in {"youtube.com", "www.youtube.com", "youtu.be", "vimeo.com"}:
        return "video"

    if domain in {"medium.com", "www.medium.com"} or domain.endswith(".medium.com"):
        return "personal_blog"

    if domain in {
        "springer.com", "link.springer.com", "nature.com", "sciencedirect.com",
        "www.sciencedirect.com", "pmc.ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov",
        "jamanetwork.com", "tandfonline.com",
        "ieee.org", "ieeexplore.ieee.org", "acm.org", "dl.acm.org",
        "aaai.org", "jmlr.org", "pnas.org"
    }:
        return "peer_reviewed_paper"

    if domain.endswith(".edu") or ".edu." in domain:
        return "academic"

    if domain in {"w3.org", "www.w3.org", "iso.org", "www.iso.org", "ietf.org", "www.ietf.org"}:
        return "standards"

    documentation_markers = (
        domain.startswith("docs."),
        domain.startswith("developer."),
        domain.startswith("developers."),
        domain.startswith("support."),
        "readthedocs" in domain,
        "documentation" in domain,
        "docs" in title
    )

    if any(documentation_markers):
        return "documentation"

    news_domains = {
        "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com", "nytimes.com",
        "cnn.com", "theguardian.com", "washingtonpost.com"
    }

    if domain in news_domains or any(
        marker in domain
        for marker in ("news.", "newsroom.")
    ):
        return "news"

    if domain.startswith("blog.") or "/blog/" in str(source.get("url", "")):
        return "corporate_blog"

    if domain.endswith(".org") and any(
        marker in combined
        for marker in ("technical report", "research report", "white paper")
    ):
        return "official"

    if any(
        marker in combined
        for marker in ("technical article", "engineering blog", "developer blog")
    ):
        return "technical_article"

    return "other"


def _tokens(value):
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value).lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _relevance_score(source, query_text, intent=None, content_status=None):
    query_tokens = _tokens(query_text)

    if not query_tokens:
        return 0.0

    title_tokens = _tokens(source.get("title", ""))
    abstract = source.get("abstract") or source.get("content", "")
    abstract_tokens = _tokens(abstract)

    title_intersection = query_tokens & title_tokens
    abstract_intersection = query_tokens & abstract_tokens

    query_weight = sum(_term_weight(t) for t in query_tokens)

    if query_weight == 0:
        return 0.0

    title_weight = sum(_term_weight(t) for t in title_intersection)
    abstract_weight = sum(_term_weight(t) for t in abstract_intersection)

    raw_score = ((title_weight / query_weight) * 0.70) + ((abstract_weight / query_weight) * 0.30)

    relevance_score = min(1.0, raw_score)

    query_lower = query_text.lower()
    source_text = (
        f"{source.get('title', '')} "
        f"{source.get('abstract', '') or source.get('content', '')}"
    ).lower()

    is_nlp_query = _is_llm_rag_question(query_text)

    if is_nlp_query:
        concept_hits = sum(
            1 for concept in _NLP_DOMAIN_CONCEPTS
            if concept in source_text
        )
        if concept_hits >= 2:
            boost = min(0.50, concept_hits * 0.15)
            relevance_score = min(1.0, relevance_score + boost)

        non_nlp_hits = sum(
            1 for indicator in _NON_NLP_DOMAIN_INDICATORS
            if indicator in source_text
        )
        non_nlp_secondary = sum(
            1 for indicator in _NON_NLP_SECONDARY_INDICATORS
            if indicator in source_text
        )

        if non_nlp_hits >= 1 and non_nlp_hits > concept_hits:
            penalty = min(
                0.60,
                non_nlp_hits * 0.20 + non_nlp_secondary * 0.05
            )
            relevance_score = max(0.0, relevance_score - penalty)

    if intent and intent in _QUERY_INTENT_CONCEPTS:
        intent_concepts = _QUERY_INTENT_CONCEPTS[intent]
        intent_hits = sum(
            1 for concept in intent_concepts
            if concept in source_text
        )

        generic_intent_terms = {"hallucination", "llm", "model", "ai", "learning", "system", "method", "approach", "study", "data", "benchmark", "retrieval", "generation"}
        specific_hits = sum(
            1 for concept in intent_concepts
            if concept in source_text and concept not in generic_intent_terms
        )

        if specific_hits >= 4:
            boost = min(0.50, specific_hits * 0.12)
            relevance_score = min(1.0, relevance_score + boost)
        elif specific_hits >= 3:
            boost = min(0.40, specific_hits * 0.10)
            relevance_score = min(1.0, relevance_score + boost)
        elif specific_hits >= 2:
            boost = min(0.25, specific_hits * 0.08)
            relevance_score = min(1.0, relevance_score + boost)
        elif intent_hits >= 3 and specific_hits >= 1:
            boost = min(0.15, specific_hits * 0.05)
            relevance_score = min(1.0, relevance_score + boost)

    if is_nlp_query:
        title_lower = str(source.get("title", "") or "").lower()
        title_mismatch = sum(
            1 for ind in _DOMAIN_MISMATCH_INDICATORS
            if ind in title_lower
        )

        llm_rag_in_title = any(
            ind in title_lower
            for ind in ("rag", "retrieval", "llm", "language model", "hallucination", "large language")
        )

        if title_mismatch >= 1:
            if llm_rag_in_title:
                penalty = min(0.30, title_mismatch * 0.15)
            else:
                penalty = min(0.70, title_mismatch * 0.35)
            relevance_score = max(0.0, relevance_score - penalty)
        else:
            body_mismatch = sum(
                1 for ind in _DOMAIN_MISMATCH_INDICATORS
                if ind in source_text
            )
            if body_mismatch >= 2:
                llm_rag_in_body = any(
                    ind in source_text
                    for ind in ("rag", "retrieval", "llm", "language model", "hallucination", "large language")
                )
                if llm_rag_in_body:
                    penalty = min(0.20, body_mismatch * 0.10)
                else:
                    penalty = min(0.40, body_mismatch * 0.15)
                relevance_score = max(0.0, relevance_score - penalty)

    if content_status == "full":
        relevance_score = min(1.0, relevance_score + 0.15)
    elif content_status == "partial":
        relevance_score = min(1.0, relevance_score + 0.10)
    elif content_status == "snippet_only":
        relevance_score = min(1.0, relevance_score + 0.05)
    elif content_status == "failed":
        relevance_score = max(0.0, relevance_score - 0.10)

    return relevance_score


def _relevance_reason(source, query_text, score, intent=None):
    query_tokens = _tokens(query_text)
    title_tokens = _tokens(source.get("title", ""))
    abstract = source.get("abstract") or source.get("content", "")
    abstract_tokens = _tokens(abstract)

    title_matches = query_tokens & title_tokens
    abstract_matches = query_tokens & abstract_tokens

    query_lower = query_text.lower()
    source_text = (
        f"{source.get('title', '')} "
        f"{source.get('abstract', '') or source.get('content', '')}"
    ).lower()

    is_nlp_query = _is_llm_rag_question(query_text)

    concept_hits = 0
    non_nlp_hits = 0
    if is_nlp_query:
        concept_hits = sum(
            1 for concept in _NLP_DOMAIN_CONCEPTS
            if concept in source_text
        )
        non_nlp_hits = sum(
            1 for indicator in _NON_NLP_DOMAIN_INDICATORS
            if indicator in source_text
        )

    intent_hits = 0
    specific_hits = 0
    if intent and intent in _QUERY_INTENT_CONCEPTS:
        intent_concepts = _QUERY_INTENT_CONCEPTS[intent]
        intent_hits = sum(
            1 for concept in intent_concepts
            if concept in source_text
        )
        generic_intent_terms = {"hallucination", "llm", "model", "ai", "learning", "system", "method", "approach", "study", "data", "benchmark", "retrieval", "generation"}
        specific_hits = sum(
            1 for concept in intent_concepts
            if concept in source_text and concept not in generic_intent_terms
        )

    title_lower = str(source.get("title", "") or "").lower()
    title_mismatch = sum(
        1 for ind in _DOMAIN_MISMATCH_INDICATORS
        if ind in title_lower
    )

    if is_nlp_query and title_mismatch >= 1:
        return "domain_mismatch_penalty"
    elif is_nlp_query and non_nlp_hits >= 2 and non_nlp_hits > concept_hits:
        return "non_nlp_domain_penalty"
    elif intent_hits >= 4 and specific_hits >= 3:
        return "strong_question_intent_match"
    elif intent_hits >= 3 and specific_hits >= 2:
        return "direct_comparison_match"
    elif intent_hits >= 2 and specific_hits >= 2:
        return "retrieval_configuration_match"
    elif intent_hits >= 2 and "failure mode" in source_text:
        return "failure_mode_match"
    elif intent_hits >= 2 and "definition" in source_text:
        return "hallucination_definition_match"
    elif score >= 0.5 and len(title_matches) >= 2:
        return "strong_title_match"
    elif score >= 0.35 and len(title_matches) >= 1:
        return "moderate_title_match"
    elif score >= 0.15 and len(abstract_matches) >= 2:
        return "abstract_match"
    elif concept_hits >= 3:
        return "strong_nlp_domain_match"
    elif concept_hits >= 2:
        return "domain_concept_match"
    elif score > 0.0:
        return "generic_overlap_only"
    else:
        return "no_significant_overlap"


def _planner_preference_score(source_type, preferred_source_types):
    if not preferred_source_types:
        return 0.5

    preferred_types = set()

    for preference in preferred_source_types:
        preference_key = str(preference).strip().lower()
        preferred_types.update(
            PREFERENCE_TO_SOURCE_TYPES.get(
                preference_key,
                set()
            )
        )

    if not preferred_types:
        return 0.5

    return 1.0 if source_type in preferred_types else 0.0


def _tavily_score(source):
    score = source.get("score")

    if score is None:
        return 0.5

    try:
        return max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        return 0.5


def score_source(source, research_question):
    """Score one source relative to one Planner question."""

    source_type = source.get(
        "source_type",
        "other"
    )

    query_texts = [
        research_question.get("question", "")
    ]

    question_id = str(
        research_question.get("id", "")
    ).strip().upper()

    for discovery in source.get("discovered_by", []):
        if discovery.get("question_id", "").upper() == question_id:
            query_texts.append(discovery.get("query_text", ""))

    best_relevance = 0.0
    best_relevance_reason = "no_query"
    intent = _detect_query_intent(
        research_question.get("question", ""),
        research_question.get("type")
    )
    content_status = str(source.get("content_status", "") or "").lower()
    for query in query_texts:
        q_score = _relevance_score(source, query, intent=intent, content_status=content_status)
        if q_score > best_relevance:
            best_relevance = q_score
            best_relevance_reason = _relevance_reason(source, query, q_score, intent=intent)

    relevance_score = best_relevance
    source_quality_score = SOURCE_QUALITY_SCORES.get(source_type, SOURCE_QUALITY_SCORES["other"])
    planner_preference_score = _planner_preference_score(
        source_type,
        research_question.get("preferred_source_types", [])
    )
    tavily_score = _tavily_score(source)

    content_status_bonus = 0.0
    if content_status == "full":
        content_status_bonus = 0.15
    elif content_status == "partial":
        content_status_bonus = 0.10
    elif content_status == "snippet_only":
        content_status_bonus = 0.05
    elif content_status == "failed":
        content_status_bonus = -0.10

    final_score = (
        relevance_score * RANKING_WEIGHTS["relevance"]
        + source_quality_score * RANKING_WEIGHTS["source_quality"]
        + planner_preference_score * RANKING_WEIGHTS["planner_preference"]
        + tavily_score * RANKING_WEIGHTS["tavily"]
        + content_status_bonus
    )

    return {
        "relevance_score": round(relevance_score, 4),
        "relevance_reason": best_relevance_reason,
        "source_quality_score": round(source_quality_score, 4),
        "planner_preference_score": round(planner_preference_score, 4),
        "tavily_score": round(tavily_score, 4),
        "content_status_bonus": round(content_status_bonus, 4),
        "final_score": round(final_score, 4)
    }


def compute_evidence_eligibility(source):
    """Determine whether a source is eligible as evidence, with an explainable reason."""
    source_type = source.get("source_type", "other")
    if source_type in EVIDENCE_ELIGIBLE_SOURCE_TYPES:
        return True, "eligible"
    reason = EVIDENCE_EXCLUSION_REASONS.get(source_type, "unsupported_source_type")
    return False, reason


def rank_sources(sources, research_questions):
    """Classify and rank all canonical sources without filtering them."""

    questions_by_id = {
        str(question.get("id", "")).strip().upper(): question
        for question in research_questions
    }

    for source in sources:
        source_type = classify_source(source)
        source["source_type"] = source_type if source_type in SOURCE_TYPES else "other"

        evidence_eligible, eligibility_reason = compute_evidence_eligibility(source)
        source["evidence_eligible"] = evidence_eligible
        source["eligibility_reason"] = eligibility_reason

        ranking_by_question = {}

        source_question_ids = {
            item.get("question_id", "").strip().upper()
            for item in source.get("discovered_by", [])
            if item.get("question_id", "").strip()
        }

        if not source_question_ids and source.get("question_id"):
            source_question_ids.add(source["question_id"].strip().upper())

        for question_id in source_question_ids:
            question = questions_by_id.get(question_id)
            if question is not None:
                ranking_by_question[question_id] = score_source(source, question)

        if ranking_by_question:
            best_question_id, best_score = max(
                ranking_by_question.items(),
                key=lambda item: (
                    item[1]["final_score"],
                    item[0]
                )
            )
            source["ranking_score"] = best_score["final_score"]
            source["ranking_question_id"] = best_question_id
        else:
            source["ranking_score"] = 0.0
            source["ranking_question_id"] = ""

        source["ranking_by_question"] = ranking_by_question

    return sorted(
        sources,
        key=lambda source: (
            -source.get("ranking_score", 0.0),
            int(source["source_id"][1:])
            if source.get("source_id", "").startswith("S")
            and source["source_id"][1:].isdigit()
            else 0
        )
    )


def print_ranking_diagnostics(
    sources,
    research_questions,
    top_n=5
):
    """Print concise per-question ranking diagnostics."""

    for question in research_questions:

        question_id = str(
            question.get(
                "id",
                ""
            )
        ).strip().upper()

        ranked_sources = [
            source
            for source in sources
            if question_id in source.get(
                "ranking_by_question",
                {}
            )
        ]

        ranked_sources.sort(
            key=lambda source: source[
                "ranking_by_question"
            ][question_id][
                "final_score"
            ],
            reverse=True
        )

        print(
            f"\n{question_id}"
        )

        print(
            "\nQuestion:"
        )

        print(
            question.get(
                "question",
                ""
            )
        )

        print(
            "\nPlanner Preferred Types:"
        )

        for preferred_type in question.get(
            "preferred_source_types",
            []
        ):

            print(
                f"- {preferred_type}"
            )

        print(
            "\nTOP RANKED SOURCES"
        )

        for position, source in enumerate(
            ranked_sources[:top_n],
            start=1
        ):

            scores = source[
                "ranking_by_question"
            ][question_id]

            provenance_types = sorted({
                discovery.get(
                    "query_type",
                    ""
                )
                for discovery in source.get(
                    "discovered_by",
                    []
                )
                if discovery.get(
                    "question_id",
                    ""
                ).strip().upper() == question_id
            })

            print(
                f"\n{position}. {source.get('source_id', '')}"
            )

            print(
                f"   title: {source.get('title', '')}"
            )

            print(
                f"   type: {source.get('source_type', 'other')}"
            )

            print(
                "   query provenance: "
                f"{', '.join(provenance_types) or 'unknown'}"
            )

            print(
                "   relevance: "
                f"{scores['relevance_score']:.4f}"
            )

            print(
                "   quality: "
                f"{scores['source_quality_score']:.4f}"
            )

            print(
                "   preference: "
                f"{scores['planner_preference_score']:.4f}"
            )

            print(
                "   tavily: "
                f"{scores['tavily_score']:.4f}"
            )

            print(
                "   final: "
                f"{scores['final_score']:.4f}"
            )

        counter_sources = [
            source
            for source in ranked_sources
            if any(
                discovery.get(
                    "query_type",
                    ""
                ).strip().lower() == "counter"
                and discovery.get(
                    "question_id",
                    ""
                ).strip().upper() == question_id
                for discovery in source.get(
                    "discovered_by",
                    []
                )
            )
        ]

        print(
            "\nCOUNTER-EVIDENCE CANDIDATES"
        )

        for source in counter_sources[:top_n]:

            score = source[
                "ranking_by_question"
            ][question_id][
                "final_score"
            ]

            print(
                f"{source.get('source_id', '')} | "
                f"type: {source.get('source_type', 'other')} | "
                f"final: {score:.4f}"
            )
