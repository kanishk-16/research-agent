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


def deduplicate_sources(
    sources
):
    """
    Merge sources with the same safe normalized URL.

    Sources are sorted by normalized_url before merging so the
    result is deterministic regardless of input order.
    """

    sorted_sources = sorted(
        sources,
        key=lambda s: (
            s.get("normalized_url")
            or normalize_url(s.get("url", ""))
            or ""
        )
    )

    unique_sources = []
    source_by_identity = {}

    for source in sorted_sources:

        normalized_url = source.get(
            "normalized_url",
            normalize_url(
                source.get(
                    "url",
                    ""
                )
            )
        )

        source["normalized_url"] = normalized_url

        if not normalized_url:
            unique_sources.append(
                source
            )

            continue

        canonical = source_by_identity.get(
            normalized_url
        )

        if canonical is None:
            source_by_identity[
                normalized_url
            ] = source

            unique_sources.append(
                source
            )

            continue

        canonical["duplicate_count"] = (
            canonical.get(
                "duplicate_count",
                0
            )
            + 1
        )

        for discovery in source.get(
            "discovered_by",
            []
        ):

            if discovery not in canonical[
                "discovered_by"
            ]:
                canonical[
                    "discovered_by"
                ].append(
                    discovery
                )

        source_url = source.get(
            "url",
            ""
        )

        if source_url and source_url != canonical.get(
            "url",
            ""
        ) and source_url not in canonical[
            "alternate_urls"
        ]:
            canonical[
                "alternate_urls"
            ].append(
                source_url
            )

        canonical_score = canonical.get(
            "score"
        )

        source_score = source.get(
            "score"
        )

        if source_score is not None and (
            canonical_score is None
            or source_score > canonical_score
        ):
            canonical[
                "score"
            ] = source_score

        if not canonical.get(
            "content",
            ""
        ) and source.get(
            "content",
            ""
        ):
            canonical[
                "content"
            ] = source[
                "content"
            ]

        if not canonical.get(
            "title",
            ""
        ) and source.get(
            "title",
            ""
        ):
            canonical[
                "title"
            ] = source[
                "title"
            ]

    return unique_sources
