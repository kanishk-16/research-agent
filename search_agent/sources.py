import re
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlsplit,
    urlunsplit
)


REMOVABLE_QUERY_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "_ga",
    "_gid",
    "msclkid",
    "mkt_tok",
    "igshid",
    "si",
    "sa",
    "dficid",
}


def normalize_url(
    url
):
    """
    Normalize safe URL identity components without
    changing the original source URL.
    """

    if not isinstance(
        url,
        str
    ):
        return ""

    url = url.strip()

    if not url:
        return ""

    parsed = urlsplit(
        url
    )

    scheme = parsed.scheme.lower()
    hostname = (
        parsed.hostname
        or ""
    ).lower()

    if not hostname:
        return url

    try:
        port = parsed.port

    except ValueError:
        return url

    netloc = hostname

    if ":" in hostname and not hostname.startswith("["):
        netloc = f"[{hostname}]"

    if port is not None and not (
        (scheme == "http" and port == 80)
        or (scheme == "https" and port == 443)
    ):
        netloc = f"{netloc}:{port}"

    path = parsed.path or "/"

    if path != "/":
        path = path.rstrip("/")

    query_parameters = []

    for key, value in parse_qsl(
        parsed.query,
        keep_blank_values=True
    ):

        if key.lower().startswith(
            "utm_"
        ) or key.lower() in REMOVABLE_QUERY_PARAMETERS:
            continue

        query_parameters.append(
            (key, value)
        )

    query_parameters.sort(key=lambda item: item[0].lower())

    query = urlencode(
        query_parameters,
        doseq=True
    )

    normalized = urlunsplit(
        (
            scheme,
            netloc,
            path,
            query,
            ""
        )
    )

    if hostname in {
        "arxiv.org",
        "export.arxiv.org"
    }:

        normalized = normalized.replace(
            "/abs/",
            "/abs/",
            1
        )

        if "/abs/" in normalized:
            prefix, identifier = normalized.split(
                "/abs/",
                1
            )

            if identifier and identifier.rsplit(
                "v",
                1
            )[-1].isdigit() and "v" in identifier:
                identifier = identifier.rsplit(
                    "v",
                    1
                )[0]

                normalized = (
                    f"{prefix}/abs/{identifier}"
                )

    return normalized


def create_source(
    source_id,
    query_record,
    result
):
    """
    Construct one structured source from a Tavily result.
    """

    question_id = str(
        query_record.get(
            "question_id",
            ""
        )
    ).strip().upper()

    query_type = str(
        query_record.get(
            "query_type",
            ""
        )
    ).strip().lower()

    if query_type not in {
        "normal",
        "counter",
        "legacy"
    }:
        raise RuntimeError(
            f"Invalid query type for {source_id}: {query_type}"
        )

    if query_type != "legacy" and not question_id:
        raise RuntimeError(
            f"Missing question ID for {source_id}."
        )

    url = result.get(
        "url",
        ""
    )

    if url is None:
        url = ""

    if not isinstance(
        url,
        str
    ):
        raise RuntimeError(
            f"Source URL must be text for {source_id}."
        )

    normalized_source_id = str(
        source_id
    ).strip().upper()

    if not normalized_source_id:
        raise RuntimeError(
            "Source ID must not be empty."
        )

    score = result.get(
        "score"
    )

    if score is not None and not isinstance(
        score,
        (int, float)
    ):
        raise RuntimeError(
            f"Source score must be numeric or None for {source_id}."
        )

    return {
        "source_id": normalized_source_id,
        "question_id": question_id,
        "query_text": str(
            query_record.get(
                "query_text",
                ""
            )
        ).strip(),
        "query_type": query_type,
        "title": str(
            result.get(
                "title",
                ""
            )
            or ""
        ).strip(),
        "url": url,
        "normalized_url": normalize_url(url),
        "content": str(
            result.get(
                "content",
                ""
            )
            or ""
        ),
        "score": score,
        "discovered_by": [
            {
                "question_id": question_id,
                "query_text": str(
                    query_record.get(
                        "query_text",
                        ""
                    )
                ).strip(),
                "query_type": query_type
            }
        ],
        "duplicate_count": 0,
        "alternate_urls": []
    }


def build_sources(
    query_record,
    results,
    starting_source_number
):
    """
    Convert one Tavily response into sequential source objects.
    """

    sources = []

    for result_index, result in enumerate(
        results,
        start=starting_source_number
    ):

        sources.append(
            create_source(
                f"S{result_index}",
                query_record,
                result
            )
        )

    return sources


def _extract_arxiv_id(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    match = re.search(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:)?([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?", text.lower())
    if match:
        return match.group(1)
    old_match = re.search(r"arxiv\.org/(?:abs|pdf)/([a-z\-]+/[0-9]{7})", text.lower())
    if old_match:
        return old_match.group(1)
    return ""


def _extract_canonical_keys(source: dict) -> list[str]:
    keys = []

    # 1. DOI
    doi = source.get("doi")
    if doi and isinstance(doi, str):
        clean_doi = doi.strip().lower()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if clean_doi.startswith(prefix):
                clean_doi = clean_doi[len(prefix):]
        if clean_doi:
            keys.append(f"doi:{clean_doi}")

    # 2. arXiv ID
    for field in ("url", "normalized_url", "doi"):
        val = source.get(field)
        if val:
            aid = _extract_arxiv_id(str(val))
            if aid:
                keys.append(f"arxiv:{aid}")
                break

    # 3. Normalized URL
    norm_url = source.get("normalized_url") or normalize_url(source.get("url", ""))
    if norm_url:
        keys.append(f"url:{norm_url}")

    # 4. Normalized Title (length >= 15)
    title = source.get("title")
    if title and isinstance(title, str):
        clean_title = re.sub(r"[^a-z0-9]", "", title.lower())
        if len(clean_title) >= 15:
            keys.append(f"title:{clean_title}")

    return keys


def deduplicate_sources(
    sources
):
    """
    Merge sources across providers using a multi-key canonical cascade:
    1. DOI
    2. arXiv ID
    3. Normalized URL
    4. Normalized alphanumeric title (>= 15 chars)

    Deterministic merging preserves richest metadata and highest scores.
    """

    sorted_sources = sorted(
        sources,
        key=lambda s: (
            s.get("source_id", ""),
            s.get("normalized_url") or normalize_url(s.get("url", "")) or ""
        )
    )

    unique_sources = []
    key_to_canonical = {}

    for source in sorted_sources:
        normalized_url = source.get(
            "normalized_url",
            normalize_url(source.get("url", ""))
        )
        source["normalized_url"] = normalized_url

        keys = _extract_canonical_keys(source)

        canonical = None
        for k in keys:
            if k in key_to_canonical:
                canonical = key_to_canonical[k]
                break

        if canonical is None:
            for k in keys:
                key_to_canonical[k] = source
            unique_sources.append(source)
            continue

        canonical["duplicate_count"] = (
            canonical.get("duplicate_count", 0) + 1
        )

        for k in keys:
            if k not in key_to_canonical:
                key_to_canonical[k] = canonical

        for discovery in source.get("discovered_by", []):
            if discovery not in canonical.get("discovered_by", []):
                canonical.setdefault("discovered_by", []).append(discovery)

        source_url = source.get("url", "")
        if source_url and source_url != canonical.get("url", ""):
            canonical.setdefault("alternate_urls", [])
            if source_url not in canonical["alternate_urls"]:
                canonical["alternate_urls"].append(source_url)

        canonical_score = canonical.get("score")
        source_score = source.get("score")
        if source_score is not None and (
            canonical_score is None or source_score > canonical_score
        ):
            canonical["score"] = source_score

        can_content = canonical.get("content", "")
        src_content = source.get("content", "")
        if len(src_content) > len(can_content):
            canonical["content"] = src_content

        can_title = canonical.get("title", "")
        src_title = source.get("title", "")
        if not can_title and src_title:
            canonical["title"] = src_title

        # Preserve and enrich academic / bibliographic metadata
        for meta_field in (
            "authors", "publication_year", "doi", "venue",
            "abstract", "tldr", "fields_of_study", "publication_types",
            "open_access_url", "is_open_access", "provider_name"
        ):
            if not canonical.get(meta_field) and source.get(meta_field):
                canonical[meta_field] = source[meta_field]

        if source.get("citation_count") is not None:
            can_cc = canonical.get("citation_count")
            if can_cc is None or source["citation_count"] > can_cc:
                canonical["citation_count"] = source["citation_count"]

        if source.get("influential_citation_count") is not None:
            can_icc = canonical.get("influential_citation_count")
            if can_icc is None or source["influential_citation_count"] > can_icc:
                canonical["influential_citation_count"] = source["influential_citation_count"]

    return unique_sources


PRIMARY_SOURCE_TYPES = {
    "peer_reviewed_paper",
    "peer-reviewed paper",
    "peer_reviewed_journal_paper",
    "peer-reviewed journal article",
    "peer-reviewed conference paper",
    "conference_paper",
    "conference_proceeding",
    "journal_article",
    "proceedings_article",
    "preprint",
    "primary_research_preprint",
    "archival preprint from reputable institution",
    "systematic_review",
    "systematic literature review",
    "official_technical_report",
    "official technical report",
    "industry research benchmark documentation",
    "academic",
    "official",
    "government",
}

PEER_REVIEWED_SOURCE_TYPES = {
    "peer_reviewed_paper",
    "peer-reviewed paper",
    "peer_reviewed_journal_paper",
    "peer-reviewed journal article",
    "peer-reviewed conference paper",
    "conference_paper",
    "conference_proceeding",
    "journal_article",
    "proceedings_article",
}


def is_primary_source_type(source_or_type) -> bool:
    """
    Check if a source or source_type string qualifies as a primary research source
    under Planner policy and academic provider classifications.
    """
    if isinstance(source_or_type, dict):
        if source_or_type.get("doi") or source_or_type.get("arxiv_id"):
            return True
        pub_types = [str(pt).lower() for pt in (source_or_type.get("publication_types") or [])]
        if any(pt in ("journalarticle", "conference", "review", "book") for pt in pub_types):
            return True
        st = str(source_or_type.get("source_type", "") or "").strip().lower()
    else:
        st = str(source_or_type or "").strip().lower()

    if not st:
        return False
    if st in PRIMARY_SOURCE_TYPES:
        return True
    st_clean = st.replace("-", "_").replace(" ", "_")
    if any(k in st_clean for k in ("peer_reviewed", "conference", "journal", "preprint", "academic", "official", "systematic")):
        return True
    return False


def is_peer_reviewed_type(source_or_type) -> bool:
    """
    Check if a source or source_type string qualifies as a peer-reviewed academic publication.
    """
    if isinstance(source_or_type, dict):
        if source_or_type.get("peer_reviewed") is True:
            return True
        pub_types = [str(pt).lower() for pt in (source_or_type.get("publication_types") or [])]
        if any(pt in ("journalarticle", "conference") for pt in pub_types):
            return True
        st = str(source_or_type.get("source_type", "") or "").strip().lower()
    else:
        st = str(source_or_type or "").strip().lower()

    if not st:
        return False
    if st in PEER_REVIEWED_SOURCE_TYPES:
        return True
    st_clean = st.replace("-", "_").replace(" ", "_")
    if "peer_reviewed" in st_clean or "journal" in st_clean or "conference" in st_clean or "proceedings" in st_clean:
        return True
    return False

