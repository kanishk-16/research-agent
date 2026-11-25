"""
search_agent/ranking/policy.py

Source classification, quality tier scoring, Planner source policy normalization,
and evidence eligibility determination.
"""

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
    "peer-reviewed paper": {"peer_reviewed_paper", "conference_paper"},
    "peer-reviewed journal article": {"peer_reviewed_paper"},
    "clinical study": {"peer_reviewed_paper"},
    "conference paper": {"conference_paper", "peer_reviewed_paper"},
    "peer-reviewed conference paper": {"conference_paper", "peer_reviewed_paper"},
    "benchmark paper": {"peer_reviewed_paper", "conference_paper"},
    "primary research preprint": {"preprint"},
    "archived primary research preprint": {"preprint"},
    "preprint": {"preprint"},
    "official technical report": {"official", "government"},
    "technical report": {"official", "government", "academic"},
    "official documentation": {"documentation", "official"},
    "technical documentation": {"documentation", "official"},
    "repository documentation": {"documentation"},
    "technical standard": {"standards"},
    "security advisory": {"official", "government"},
    "government dataset": {"government"},
    "government report": {"government"},
    "academic research paper": {"peer_reviewed_paper", "academic"},
    "research paper": {"peer_reviewed_paper", "academic"},
    "industry research publication": {"technical_article", "official"},
    "technical blog": {"corporate_blog", "personal_blog"},
    "workshop notes": {"conference_paper", "academic"},
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
    provider_name = str(source.get("provider_name", "") or "").lower()
    venue = str(source.get("venue", "") or "").lower()
    doi = str(source.get("doi", "") or "").lower()
    url = str(source.get("url", "") or "").lower()
    title = str(source.get("title", "") or "").lower()
    pub_types = [str(pt).lower() for pt in source.get("publication_types", []) or []]

    if provider_name in ("openalex", "semanticscholar", "europepmc"):
        if (
            "systematic review" in title
            or "meta-analysis" in title
            or ("review" in pub_types and any(w in title for w in ("systematic", "survey", "overview", "meta-analysis")))
        ):
            return "systematic_review"

        if (
            "arxiv.org" in url
            or "arxiv" in venue
            or "biorxiv" in venue
            or "medrxiv" in venue
            or "preprint" in pub_types
            or "/10.48550/" in doi
        ):
            return "preprint"

        conference_indicators = (
            "conference", "proceedings", "symposium", "workshop",
            "acl", "emnlp", "naacl", "aaai", "ijcai",
            "neurips", "nips", "icml", "iclr",
            "cvpr", "iccv", "eccv", "icassp",
            "colt", "kdd", "sigir", "www", "chi",
        )
        journal_indicators = (
            "journal", "transactions", "letters",
            "review", "annals", "bulletin", "nature",
            "science", "ieee", "acm", "springer", "elsevier"
        )

        if "conference" in pub_types or any(ind in venue for ind in conference_indicators):
            return "conference_paper"
        elif "journalarticle" in pub_types or any(ind in venue for ind in journal_indicators):
            return "peer_reviewed_paper"
        elif doi and "/10.48550/" not in doi:
            return "peer_reviewed_paper"
        elif venue:
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

    if domain in news_domains or any(marker in domain for marker in ("news.", "newsroom.")):
        return "news"

    if domain.startswith("blog.") or "/blog/" in str(source.get("url", "")):
        return "corporate_blog"

    if domain.endswith(".org") and any(
        marker in combined for marker in ("technical report", "research report", "white paper")
    ):
        return "official"

    if any(
        marker in combined for marker in ("technical article", "engineering blog", "developer blog")
    ):
        return "technical_article"

    return "other"


def _planner_preference_score(source_type, preferred_source_types):
    if not preferred_source_types:
        return 0.5

    preferred_types = set()
    for preference in preferred_source_types:
        preference_key = str(preference).strip().lower()
        preferred_types.update(PREFERENCE_TO_SOURCE_TYPES.get(preference_key, set()))

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


def _normalize_policy_type_to_source_types(pref_str):
    clean = str(pref_str).strip().lower()
    if clean in PREFERENCE_TO_SOURCE_TYPES:
        return set(PREFERENCE_TO_SOURCE_TYPES[clean])
    underscore = clean.replace(" ", "_").replace("-", "_")
    if underscore in SOURCE_TYPES:
        return {underscore}
    if "blog" in clean:
        return {"corporate_blog", "personal_blog"}
    return {clean, underscore}


def _build_policy_quality_scores(source_policy):
    if not source_policy or not isinstance(source_policy, dict):
        return SOURCE_QUALITY_SCORES

    custom_scores = dict(SOURCE_QUALITY_SCORES)
    tiers = source_policy.get("tiers", [])
    for tier_info in tiers:
        tier_num = tier_info.get("tier", 3)
        role = str(tier_info.get("role", "evidence")).lower()
        types = tier_info.get("source_types", [])

        if tier_num == 1:
            base_score = 0.95
        elif tier_num == 2:
            base_score = 0.85
        else:
            base_score = 0.65 if role == "evidence" else 0.50

        for t in types:
            mapped_types = _normalize_policy_type_to_source_types(t)
            for mt in mapped_types:
                if mt in SOURCE_TYPES:
                    custom_scores[mt] = base_score

    return custom_scores


def compute_evidence_eligibility(source, source_policy=None):
    """Determine whether a source is eligible as evidence, with an explainable reason."""
    if source.get("content_status") == "failed" and not source.get("abstract") and not source.get("content"):
        return False, "content_extraction_failed"

    source_type = source.get("source_type", "other")

    if source_policy and isinstance(source_policy, dict):
        discovery_types = {str(t).lower().strip() for t in source_policy.get("discovery_only_source_types", [])}
        for disc in discovery_types:
            mapped = _normalize_policy_type_to_source_types(disc)
            if source_type in mapped:
                return False, "policy_discovery_only"

        eligible_types = {str(t).lower().strip() for t in source_policy.get("evidence_eligible_source_types", [])}
        for elig in eligible_types:
            mapped = _normalize_policy_type_to_source_types(elig)
            if source_type in mapped:
                return True, "eligible"

    if source_type in EVIDENCE_ELIGIBLE_SOURCE_TYPES:
        return True, "eligible"
    reason = EVIDENCE_EXCLUSION_REASONS.get(source_type, "unsupported_source_type")
    return False, reason
