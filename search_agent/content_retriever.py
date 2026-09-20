import re
import urllib.request
import urllib.error


def _fetch_fallback_content(url: str, timeout: int = 5) -> str:
    """
    Safely fetch HTML/text content directly via HTTP as fallback when Tavily extract fails.
    Strips HTML tags to extract raw text content.
    """
    if not url or not isinstance(url, str):
        return ""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            }
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            html = response.read().decode("utf-8", errors="ignore")
            # Remove scripts, styles
            clean_text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
            # Remove tags
            clean_text = re.sub(r"<[^>]+>", " ", clean_text)
            # Collapse whitespace
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
            return clean_text
    except Exception:
        return ""


def retrieve_selected_sources(
    tavily_client,
    unique_selected_sources
):
    """
    Perform deep/full content retrieval for selected canonical sources using Tavily extract,
    with fallback retrieval via Open Access URLs, DOIs, and direct repository HTTP copies.
    Modifies sources in place and returns retrieval summary stats.
    """

    if not unique_selected_sources:
        return {
            "attempts": 0,
            "successes": 0,
            "partials": 0,
            "snippet_only": 0,
            "failures": 0,
            "fallbacks_used": 0
        }

    # Gather URLs of unique selected sources that have a non-empty URL
    url_to_sources = {}
    urls_to_extract = []

    for source in unique_selected_sources:
        source["retrieval_attempted"] = True
        source["content_status"] = "snippet_only"
        source["retrieved_content"] = ""
        source["retrieval_error"] = None
        source["fallback_retrieval_used"] = False

        url = source.get("url", "").strip()
        if url:
            if url not in url_to_sources:
                url_to_sources[url] = []
                urls_to_extract.append(url)
            url_to_sources[url].append(source)
        else:
            source["content_status"] = "failed"
            source["retrieval_error"] = "Empty URL"

    if not urls_to_extract:
        return {
            "attempts": len(unique_selected_sources),
            "successes": 0,
            "partials": 0,
            "snippet_only": len(unique_selected_sources),
            "failures": 0,
            "fallbacks_used": 0
        }

    # Perform batch extraction call
    extracted_results_by_url = {}
    failed_urls_err = {}

    import time

    response = None
    last_err = None
    for attempt in range(3):
        try:
            response = tavily_client.extract(
                urls=urls_to_extract,
                extract_depth="basic"
            )
            break
        except Exception as exc:
            last_err = str(exc)
            if attempt < 2:
                time.sleep(3 * (attempt + 1))

    if response and isinstance(response, dict):
        results = response.get("results", [])
        failed_results = response.get("failed_results", [])

        for res in results:
            url = res.get("url", "")
            raw_content = str(res.get("raw_content", "") or res.get("content", "") or "").strip()
            extracted_results_by_url[url] = raw_content

        for fail in failed_results:
            url = fail.get("url", "")
            err = fail.get("error", "Extraction failed")
            failed_urls_err[url] = err
    else:
        err_msg = last_err or "Extraction API timed out"
        for url in urls_to_extract:
            failed_urls_err[url] = f"Extraction API exception: {err_msg}"

    # Update metadata on each source with Tavily results
    for source in unique_selected_sources:
        url = source.get("url", "").strip()
        if not url:
            continue

        if url in extracted_results_by_url:
            retrieved_text = extracted_results_by_url[url]
            if retrieved_text:
                source["retrieved_content"] = retrieved_text
                snippet_len = len(source.get("content", ""))
                if len(retrieved_text) > snippet_len:
                    source["content_status"] = "full"
                else:
                    source["content_status"] = "partial"
            else:
                source["content_status"] = "snippet_only"
                source["retrieved_content"] = source.get("content", "")
        elif url in failed_urls_err:
            source["content_status"] = "failed"
            source["retrieval_error"] = failed_urls_err[url]
            source["retrieved_content"] = source.get("content", "")
        else:
            source["content_status"] = "snippet_only"
            source["retrieved_content"] = source.get("content", "")

    # Multi-path fallback retrieval for failed or snippet-only sources
    fallbacks_used = 0
    for source in unique_selected_sources:
        if source.get("content_status") in ("failed", "snippet_only") or len(source.get("retrieved_content", "")) < 300:
            candidate_urls = []
            # 1. Open Access URL from OpenAlex / Semantic Scholar
            if source.get("open_access_url"):
                candidate_urls.append(source["open_access_url"])
            # 2. DOI redirect URL
            doi = source.get("doi")
            if doi and isinstance(doi, str):
                clean_doi = doi.strip()
                if not clean_doi.startswith("http"):
                    clean_doi = f"https://doi.org/{clean_doi}"
                candidate_urls.append(clean_doi)
            # 3. Alternate URLs from deduplication
            for alt in source.get("alternate_urls", []):
                if alt and alt not in candidate_urls and alt != source.get("url"):
                    candidate_urls.append(alt)
            # 4. Direct URL
            if source.get("url") and source.get("url") not in candidate_urls:
                candidate_urls.append(source["url"])

            for cand_url in candidate_urls:
                fallback_text = _fetch_fallback_content(cand_url, timeout=4)
                if len(fallback_text) > 400:
                    source["retrieved_content"] = fallback_text
                    source["content_status"] = "full" if len(fallback_text) > 2000 else "partial"
                    source["retrieval_error"] = None
                    source["fallback_retrieval_used"] = True
                    source["fallback_url"] = cand_url
                    fallbacks_used += 1
                    break

    # Re-tally statistics
    successes = sum(1 for s in unique_selected_sources if s.get("content_status") == "full")
    partials = sum(1 for s in unique_selected_sources if s.get("content_status") == "partial")
    snippet_only = sum(1 for s in unique_selected_sources if s.get("content_status") == "snippet_only")
    failures = sum(1 for s in unique_selected_sources if s.get("content_status") == "failed")

    return {
        "attempts": len(unique_selected_sources),
        "successes": successes,
        "partials": partials,
        "snippet_only": snippet_only,
        "failures": failures,
        "fallbacks_used": fallbacks_used
    }
