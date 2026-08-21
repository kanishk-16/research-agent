def retrieve_selected_sources(
    tavily_client,
    unique_selected_sources
):
    """
    Perform deep/full content retrieval for selected canonical sources using Tavily extract.
    Modifies sources in place and returns retrieval summary stats.
    """

    if not unique_selected_sources:
        return {
            "attempts": 0,
            "successes": 0,
            "partials": 0,
            "snippet_only": 0,
            "failures": 0
        }

    # Gather URLs of unique selected sources that have a non-empty URL
    url_to_sources = {}
    urls_to_extract = []

    for source in unique_selected_sources:
        # Initialize default retrieval metadata fields
        source["retrieval_attempted"] = True
        source["content_status"] = "snippet_only"
        source["retrieved_content"] = ""
        source["retrieval_error"] = None

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
            "failures": 0
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

    # Update metadata on each source
    successes = 0
    partials = 0
    snippet_only = 0
    failures = 0

    for source in unique_selected_sources:
        url = source.get("url", "").strip()
        if not url:
            failures += 1
            continue

        if url in extracted_results_by_url:
            retrieved_text = extracted_results_by_url[url]
            if retrieved_text:
                source["retrieved_content"] = retrieved_text
                # Determine status based on text length relative to original search snippet
                snippet_len = len(source.get("content", ""))
                if len(retrieved_text) > snippet_len:
                    source["content_status"] = "full"
                    successes += 1
                else:
                    source["content_status"] = "partial"
                    partials += 1
            else:
                source["content_status"] = "snippet_only"
                source["retrieved_content"] = source.get("content", "")
                snippet_only += 1
        elif url in failed_urls_err:
            source["content_status"] = "failed"
            source["retrieval_error"] = failed_urls_err[url]
            source["retrieved_content"] = source.get("content", "")
            failures += 1
        else:
            source["content_status"] = "snippet_only"
            source["retrieved_content"] = source.get("content", "")
            snippet_only += 1

    return {
        "attempts": len(unique_selected_sources),
        "successes": successes,
        "partials": partials,
        "snippet_only": snippet_only,
        "failures": failures
    }
