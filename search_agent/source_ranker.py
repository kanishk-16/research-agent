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
        "jamanetwork.com", "tandfonline.com"
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

    if any(
        marker in combined
        for marker in ("systematic review", "systematic survey", "meta-analysis")
    ):
        return "systematic_review"

    if domain.startswith("blog.") or "/blog/" in str(source.get("url", "")):
        return "corporate_blog"

    if domain.endswith(".org") and any(
        marker in combined
        for marker in ("technical report", "research report", "white paper")
    ):
        return "official"

    if domain.endswith(".org"):
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


def _relevance_score(source, query_text):
    query_tokens = _tokens(query_text)

    if not query_tokens:
        return 0.0

    title_tokens = _tokens(source.get("title", ""))
    content_tokens = _tokens(source.get("content", ""))
    title_overlap = len(query_tokens & title_tokens) / len(query_tokens)
    content_overlap = len(query_tokens & content_tokens) / len(query_tokens)
    return min(1.0, (title_overlap * 0.70) + (content_overlap * 0.30))


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

    relevance_score = max(
        (_relevance_score(source, query) for query in query_texts),
        default=0.0
    )
    source_quality_score = SOURCE_QUALITY_SCORES.get(source_type, SOURCE_QUALITY_SCORES["other"])
    planner_preference_score = _planner_preference_score(
        source_type,
        research_question.get("preferred_source_types", [])
    )
    tavily_score = _tavily_score(source)
    final_score = (
        relevance_score * RANKING_WEIGHTS["relevance"]
        + source_quality_score * RANKING_WEIGHTS["source_quality"]
        + planner_preference_score * RANKING_WEIGHTS["planner_preference"]
        + tavily_score * RANKING_WEIGHTS["tavily"]
    )

    return {
        "relevance_score": round(relevance_score, 4),
        "source_quality_score": round(source_quality_score, 4),
        "planner_preference_score": round(planner_preference_score, 4),
        "tavily_score": round(tavily_score, 4),
        "final_score": round(final_score, 4)
    }


def rank_sources(sources, research_questions):
    """Classify and rank all canonical sources without filtering them."""

    questions_by_id = {
        str(question.get("id", "")).strip().upper(): question
        for question in research_questions
    }

    for source in sources:
        source_type = classify_source(source)
        source["source_type"] = source_type if source_type in SOURCE_TYPES else "other"
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
