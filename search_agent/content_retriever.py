import re
import urllib.request
import urllib.error
import io
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List


def _compute_backoff_delay(attempt: int, base_delay: float = 0.4, max_delay: float = 6.0) -> float:
    """Compute exponential backoff delay with jitter."""
    cap = min(max_delay, base_delay * (2 ** attempt))
    return round(random.uniform(base_delay * 0.5, cap), 3)


def download_and_parse_pdfs_parallel(
    urls: List[str],
    max_workers: int = 6,
    timeout: int = 6
) -> Dict[str, str]:
    """
    Download and parse multiple candidate PDF / HTML full-text URLs concurrently
    using a ThreadPoolExecutor. Returns {url: extracted_text}.
    """
    unique_urls = [u for u in dict.fromkeys(urls or []) if u and isinstance(u, str)]
    if not unique_urls:
        return {}
    if len(unique_urls) == 1:
        return {unique_urls[0]: _fetch_fallback_content(unique_urls[0], timeout=timeout)}

    workers = min(max_workers, len(unique_urls))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda u: _fetch_fallback_content(u, timeout=timeout), unique_urls))
    return dict(zip(unique_urls, results))


def _extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = 15) -> str:
    """
    Extract readable text from PDF binary bytes using pypdf.
    Returns extracted text string, or empty string on failure.
    """
    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF"):
        return ""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages[:max_pages]:
            t = page.extract_text() or ""
            if t.strip():
                pages_text.append(t.strip())
        return "\n\n".join(pages_text).strip()
    except Exception:
        return ""


def _extract_arxiv_id(url: str) -> str:
    """Extract arXiv ID from arXiv/ar5iv URL or DOI (e.g., '2402.17840' or '2402.17840v1')."""
    if not url:
        return ""
    m = re.search(
        r'(?:(?:arxiv\.org|ar5iv\.labs\.arxiv\.org)/(?:abs|html|pdf)/|arxiv:|10\.48550/arxiv\.|/arxiv\.)([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)',
        url,
        re.I,
    )
    if m:
        return m.group(1)
    return ""


def build_ar5iv_html_url(arxiv_id: str) -> str:
    """Construct clean pre-rendered HTML5 ar5iv URL for an arXiv identifier."""
    clean_id = str(arxiv_id or "").strip()
    if not clean_id:
        return ""
    return f"https://ar5iv.labs.arxiv.org/html/{clean_id}"


def resolve_unpaywall_pdf_url(
    doi: str,
    email: str = "ara-research-agent@example.org",
    timeout: int = 4,
) -> str:
    """
    Resolve a direct open-access legal PDF or full-text URL from any DOI using the Unpaywall REST API
    (https://api.unpaywall.org/v2/<doi>?email=...), bypassing paywalled publisher landing pages.
    """
    if not doi or not isinstance(doi, str):
        return ""
    clean_doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi.strip(), flags=re.I).strip()
    if not clean_doi or not clean_doi.startswith("10."):
        return ""

    import json
    import urllib.parse
    api_url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(clean_doi, safe='/')}?email={urllib.parse.quote(email)}"
    try:
        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": "ARA-ResearchAgent/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return ""
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            best_loc = data.get("best_oa_location") or {}
            pdf_url = best_loc.get("url_for_pdf") or best_loc.get("url")
            if pdf_url and isinstance(pdf_url, str):
                return pdf_url.strip()
            for loc in data.get("oa_locations") or []:
                if isinstance(loc, dict):
                    u = loc.get("url_for_pdf") or loc.get("url")
                    if u:
                        return str(u).strip()
    except Exception:
        return ""
    return ""


def _fetch_fallback_content(url: str, timeout: int = 6) -> str:
    """
    Safely fetch full-text content directly via HTTP as fallback when Tavily extract fails or yields only snippets.
    Supports ar5iv pre-rendered HTML5, arXiv HTML papers, direct PDF extraction via pypdf, and HTML text extraction.
    """
    if not url or not isinstance(url, str):
        return ""

    # Check if URL is an arXiv paper -> try ar5iv HTML5, then arXiv HTML, then arXiv PDF
    arxiv_id = _extract_arxiv_id(url)
    if arxiv_id:
        # 1. Try ar5iv pre-rendered HTML5 representation (preserves tables, formulas, and clean text)
        ar5iv_url = build_ar5iv_html_url(arxiv_id)
        ar5iv_text = _fetch_single_url_content(ar5iv_url, timeout=timeout)
        if len(ar5iv_text) >= 2000:
            return ar5iv_text

        # 2. Try native arXiv HTML representation
        html_url = f"https://arxiv.org/html/{arxiv_id}"
        html_text = _fetch_single_url_content(html_url, timeout=timeout)
        if len(html_text) >= 2000:
            return html_text

        # 3. Try arXiv PDF representation (full text via pypdf)
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_text = _fetch_single_url_content(pdf_url, timeout=timeout)
        if len(pdf_text) >= 1500:
            return pdf_text

    return _fetch_single_url_content(url, timeout=timeout)


def _fetch_single_url_content(url: str, timeout: int = 6) -> str:
    """Fetch and decode either PDF or HTML text from a single URL."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml,application/pdf;q=0.9,*/*;q=0.8"
            }
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            raw_bytes = response.read()

            # Handle PDF content
            if "application/pdf" in content_type or url.lower().endswith(".pdf") or raw_bytes.startswith(b"%PDF"):
                pdf_text = _extract_text_from_pdf_bytes(raw_bytes)
                if pdf_text:
                    return pdf_text

            # Handle HTML / text content
            html = raw_bytes.decode("utf-8", errors="ignore")
            # If this is an arXiv abstract page, extract abstract specifically if present
            abs_match = re.search(r'<blockquote class="abstract[^"]*">(.*?)</blockquote>', html, flags=re.DOTALL | re.IGNORECASE)
            if abs_match:
                clean_abs = re.sub(r"<[^>]+>", " ", abs_match.group(1))
                clean_abs = re.sub(r"\s+", " ", clean_abs).strip()
            else:
                clean_abs = ""

            # Clean full HTML text
            clean_text = re.sub(r"<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
            clean_text = re.sub(r"<[^>]+>", " ", clean_text)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()

            if clean_abs and len(clean_text) < 1000:
                return f"Abstract: {clean_abs}\n\n{clean_text}"
            return clean_text
    except Exception:
        return ""


def _assess_content_status(text: str) -> str:
    """Accurately determine content completeness status."""
    if not text or not text.strip():
        return "failed"
    text_len = len(text.strip())
    has_empirical = bool(re.search(r"\b(results|experiments|evaluation|findings|method|dataset|benchmark|accuracy|table)\b", text, re.I))
    if text_len >= 3500 or (text_len >= 1800 and has_empirical):
        return "full"
    elif text_len >= 600:
        return "partial"
    return "snippet_only"


def retrieve_selected_sources(
    tavily_client,
    unique_selected_sources
):
    """
    Perform deep/full content retrieval for selected canonical sources using Tavily extract,
    with fallback retrieval via Open Access URLs, DOIs, arXiv HTML/PDF, and direct HTTP copies.
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
        existing_status = source.get("content_status")
        existing_content = source.get("retrieved_content") or ""
        # If source already has full or partial content, preserve it and skip re-extracting
        if existing_status in ("full", "partial") and len(existing_content.strip()) >= 600:
            source["retrieval_attempted"] = True
            continue

        source["retrieval_attempted"] = True
        if not source.get("content_status"):
            source["content_status"] = "snippet_only"
        if not source.get("retrieved_content"):
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
            "fallbacks_used": 0
        }

    # Perform batch extraction call
    extracted_results_by_url = {}
    failed_urls_err = {}

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
                time.sleep(_compute_backoff_delay(attempt))

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
            if retrieved_text and len(retrieved_text) > len(source.get("retrieved_content", "")):
                source["retrieved_content"] = retrieved_text
                source["content_status"] = _assess_content_status(retrieved_text)
            elif not source.get("retrieved_content"):
                source["content_status"] = "snippet_only"
                source["retrieved_content"] = source.get("content", "")
        elif url in failed_urls_err:
            if not source.get("retrieved_content"):
                source["content_status"] = "failed"
                source["retrieval_error"] = failed_urls_err[url]
                source["retrieved_content"] = source.get("content", "")
        else:
            if not source.get("retrieved_content"):
                source["content_status"] = "snippet_only"
                source["retrieved_content"] = source.get("content", "")

    # Multi-path fallback retrieval for sources lacking full empirical content
    from concurrent.futures import ThreadPoolExecutor

    def _fallback_one_source(source):
        local_fallbacks = 0
        current_status = source.get("content_status", "snippet_only")
        current_len = len(source.get("retrieved_content", ""))

        # Trigger fallback if not full, or if content is under 2000 chars
        if current_status != "full" or current_len < 2000:
            candidate_urls = []
            
            # 1. Open Access URL from OpenAlex / Semantic Scholar
            oa_url = source.get("open_access_url")
            if oa_url:
                candidate_urls.append(oa_url)
                
            # 2. ArXiv / ar5iv HTML5 transformations
            src_url = source.get("url", "")
            arxiv_id = _extract_arxiv_id(src_url) or _extract_arxiv_id(oa_url or "") or _extract_arxiv_id(source.get("doi") or "")
            if arxiv_id:
                candidate_urls.append(build_ar5iv_html_url(arxiv_id))
                candidate_urls.append(f"https://arxiv.org/html/{arxiv_id}")
                candidate_urls.append(f"https://arxiv.org/pdf/{arxiv_id}.pdf")
                candidate_urls.append(f"https://arxiv.org/abs/{arxiv_id}")

            # 3. Unpaywall Open-Access PDF resolution & DOI redirect URL
            doi = source.get("doi")
            if doi and isinstance(doi, str):
                unpaywall_url = resolve_unpaywall_pdf_url(doi, timeout=3)
                if unpaywall_url and unpaywall_url not in candidate_urls:
                    candidate_urls.insert(0, unpaywall_url)
                clean_doi = doi.strip()
                if not clean_doi.startswith("http"):
                    clean_doi = f"https://doi.org/{clean_doi}"
                candidate_urls.append(clean_doi)

            # 4. Alternate URLs from deduplication
            for alt in source.get("alternate_urls", []):
                if alt and alt not in candidate_urls and alt != src_url:
                    candidate_urls.append(alt)

            # 5. Direct URL
            if src_url and src_url not in candidate_urls:
                candidate_urls.append(src_url)

            best_text = source.get("retrieved_content", "")
            for cand_url in candidate_urls[:3]:
                fallback_text = _fetch_fallback_content(cand_url, timeout=3)
                if len(fallback_text) > max(len(best_text), 400):
                    best_text = fallback_text
                    source["retrieved_content"] = best_text
                    source["content_status"] = _assess_content_status(best_text)
                    source["retrieval_error"] = None
                    source["fallback_retrieval_used"] = True
                    source["fallback_url"] = cand_url
                    local_fallbacks += 1
                    if source["content_status"] == "full":
                        break

            # 6. If extraction still failed or under 1000 chars, resolve academic paper via title search
            if (source.get("content_status") == "failed" or len(best_text) < 1000) and tavily_client and source.get("title"):
                title_clean = re.sub(r"[^\w\s]", " ", source["title"]).strip()
                if len(title_clean) > 10:
                    try:
                        res_search = tavily_client.search(f'"{title_clean}" (arxiv OR aclanthology OR filetype:pdf)', max_results=2)
                        for r in res_search.get("results", []):
                            r_url = r.get("url", "")
                            if r_url and r_url != src_url:
                                f_text = _fetch_fallback_content(r_url, timeout=3)
                                if len(f_text) > max(len(best_text), 400):
                                    best_text = f_text
                                    source["retrieved_content"] = best_text
                                    source["content_status"] = _assess_content_status(best_text)
                                    source["retrieval_error"] = None
                                    source["fallback_retrieval_used"] = True
                                    source["fallback_url"] = r_url
                                    local_fallbacks += 1
                                    if source["content_status"] == "full":
                                        break
                    except Exception:
                        pass

            # 7. Final status reconciliation: if abstract snippet exists, never leave as failed
            snippet_text = source.get("content", "").strip()
            if source.get("content_status") == "failed" and len(snippet_text) >= 100:
                source["content_status"] = "snippet_only"
                source["retrieved_content"] = snippet_text
                source["retrieval_error"] = None
        return local_fallbacks

    if len(unique_selected_sources) > 1:
        with ThreadPoolExecutor(max_workers=5) as pool:
            fallbacks_used = sum(pool.map(_fallback_one_source, unique_selected_sources))
    else:
        fallbacks_used = sum(_fallback_one_source(s) for s in unique_selected_sources)

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
