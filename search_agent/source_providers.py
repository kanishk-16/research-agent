import json
import math
import os
import time
import urllib.request
import urllib.error
import urllib.parse

from .sources import normalize_url


OPENALEX_API_BASE = "https://api.openalex.org/works"
OPENALEX_PER_PAGE = 25
OPENALEX_MAX_PAGES = 3
OPENALEX_DELAY_SECONDS = 0.5



class SourceProvider:
    """Base class for academic source providers."""

    name = "base"

    def search(self, query_text, question_id=None, **kwargs):
        """Return a list of source dicts compatible with the Phase 2 source schema."""
        raise NotImplementedError


class OpenAlexProvider(SourceProvider):
    name = "openalex"

    def __init__(self, mailto=None, per_page=None, max_pages=None, delay=None):
        self.mailto = mailto
        self.per_page = per_page or OPENALEX_PER_PAGE
        self.max_pages = max_pages or OPENALEX_MAX_PAGES
        self.delay = delay if delay is not None else OPENALEX_DELAY_SECONDS

    def search(self, query_text, question_id=None, **kwargs):
        if not query_text or not str(query_text).strip():
            return []

        query_text = str(query_text).strip()
        question_id = str(question_id or "").strip().upper()
        all_results = []

        for page in range(1, self.max_pages + 1):
            page_results = self._fetch_page(
                query_text,
                page,
                question_id=question_id,
            )
            all_results.extend(page_results)
            if not page_results or len(page_results) < self.per_page:
                break
            if page < self.max_pages:
                time.sleep(self.delay)

        return all_results

    def _fetch_page(self, query_text, page, question_id=None):
        params = {
            "search": query_text,
            "per-page": self.per_page,
            "page": page,
        }
        url = f"{OPENALEX_API_BASE}?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(url, method="GET")
        req.add_header(
            "User-Agent",
            "research-agent/1.0 (https://github.com/your-repo)"
        )
        if self.mailto:
            req.add_header("mailto", self.mailto)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status != 200:
                    print(
                        f"  [Warning] OpenAlex returned HTTP {resp.status}"
                    )
                    return []
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
        except urllib.error.HTTPError as exc:
            print(
                f"  [Warning] OpenAlex HTTP error {exc.code}: "
                f"{exc.reason}"
            )
            return []
        except urllib.error.URLError as exc:
            print(f"  [Warning] OpenAlex request failed: {exc.reason}")
            return []
        except (TimeoutError, OSError) as exc:
            print(f"  [Warning] OpenAlex network error: {exc}")
            return []
        except (json.JSONDecodeError, UnicodeDecodeError):
            print("  [Warning] OpenAlex returned malformed response")
            return []
        except Exception as exc:
            print(f"  [Warning] OpenAlex unexpected error: {exc}")
            return []

        raw_results = data.get("results")
        if not isinstance(raw_results, list):
            return []

        sources = []
        for item in raw_results:
            source = self._convert(item, query_text, question_id)
            if source is not None:
                sources.append(source)

        return sources

    def _convert(self, item, query_text, question_id):
        openalex_id = item.get("id", "")
        stable_id = openalex_id
        if stable_id and "/" in stable_id:
            stable_id = stable_id.rstrip("/").split("/")[-1]

        if not stable_id:
            return None

        source_id = f"OA-{stable_id}"

        title = str(item.get("title", "") or "").strip()
        if not title:
            return None

        abstract = _reconstruct_abstract(
            item.get("abstract_inverted_index")
        )

        content = abstract if abstract else ""

        url = _best_url(item)

        authorships = item.get("authorships", []) or []
        authors = []
        for authorship in authorships:
            author = authorship.get("author", {}) or {}
            name = str(author.get("display_name", "") or "").strip()
            if name and name not in authors:
                authors.append(name)

        oa_info = item.get("open_access", {}) or {}
        oa_url = oa_info.get("oa_url") or oa_info.get("url")
        is_oa = oa_info.get("is_oa")

        cited_by = item.get("cited_by_count")
        citation_count = int(cited_by) if cited_by is not None else None

        doi = item.get("doi")
        if doi and not str(doi).startswith("https://"):
            doi = f"https://doi.org/{doi}"

        venue_info = item.get("host_venue", {}) or {}
        venue_name = str(venue_info.get("display_name", "") or "").strip()

        pub_year = item.get("publication_year")
        publication_year = int(pub_year) if pub_year is not None else None

        score = None
        if citation_count is not None:
            score = round(min(0.95, max(0.50, 0.50 + 0.15 * math.log10(citation_count + 1))), 4)

        return {
            "source_id": source_id,
            "question_id": question_id,
            "query_text": query_text,
            "query_type": "normal",
            "title": title,
            "url": url,
            "normalized_url": normalize_url(url),
            "content": content,
            "score": score,
            "discovered_by": [
                {
                    "question_id": question_id,
                    "query_text": query_text,
                    "query_type": "normal",
                }
            ],
            "duplicate_count": 0,
            "alternate_urls": [],
            "authors": authors,
            "publication_year": publication_year,
            "doi": doi,
            "venue": venue_name,
            "citation_count": citation_count,
            "abstract": abstract,
            "open_access_url": oa_url,
            "is_open_access": (
                bool(is_oa) if is_oa is not None else None
            ),
            "provider_name": self.name,
            "provider_query": query_text,
        }


def _reconstruct_abstract(inverted_index):
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    positions = []
    for word, pos_list in inverted_index.items():
        if not isinstance(pos_list, list):
            continue
        for pos in pos_list:
            try:
                positions.append((int(pos), str(word)))
            except (TypeError, ValueError):
                continue
    positions.sort()
    return " ".join(word for _, word in positions)


def _best_url(item):
    doi = item.get("doi")
    if doi:
        doi = str(doi).strip()
        if not doi.startswith("https://"):
            doi = f"https://doi.org/{doi}"
        return doi

    primary = item.get("primary_location", {}) or {}
    landing = primary.get("landing_page_url")
    if landing:
        return str(landing).strip()

    oa = item.get("open_access", {}) or {}
    oa_url = oa.get("oa_url") or oa.get("url")
    if oa_url:
        return str(oa_url).strip()

    openalex_id = item.get("id", "")
    if openalex_id:
        return str(openalex_id).strip()

    return ""



class SemanticScholarProvider(SourceProvider):
    """
    Dual-Mode Semantic Scholar Provider:
    - Mode A (Authenticated): Uses Semantic Scholar Graph API if SEMANTIC_SCHOLAR_API_KEY is available.
    - Mode B (Autonomous Web Discovery): Uses Tavily site:semanticscholar.org discovery if no API key is set.
      This eliminates HTTP 429 errors while discovering genuine Semantic Scholar papers.
    """
    name = "semanticscholar"

    def __init__(self, api_key=None, tavily_client=None, limit=10):
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.tavily_client = tavily_client
        self.limit = limit

    def search(self, query_text, question_id=None, **kwargs):
        if not query_text or not str(query_text).strip():
            return []

        query_text = str(query_text).strip()
        question_id = str(question_id or "").strip().upper()

        if self.api_key:
            return self._search_graph_api(query_text, question_id)
        else:
            return self._search_via_web_discovery(query_text, question_id)

    def _search_via_web_discovery(self, query_text, question_id):
        client = self.tavily_client
        if client is None:
            tavily_key = os.getenv("TAVILY_API_KEY")
            if tavily_key:
                try:
                    from tavily import TavilyClient
                    client = TavilyClient(api_key=tavily_key)
                    self.tavily_client = client
                except Exception:
                    return []
            else:
                return []

        target_query = f"site:semanticscholar.org/paper {query_text}"
        try:
            res = client.search(target_query, max_results=self.limit)
        except Exception as exc:
            print(f"  [Warning] Semantic Scholar web discovery search failed: {exc}")
            return []

        raw_results = res.get("results", []) if isinstance(res, dict) else []
        sources = []

        for item in raw_results:
            url = str(item.get("url", "") or "").strip()
            if "semanticscholar.org/paper" not in url:
                continue

            raw_title = str(item.get("title", "") or "").strip()
            title = (
                raw_title.replace(" | Semantic Scholar", "")
                .replace("[PDF]", "")
                .strip()
            )
            if not title:
                continue

            parts = url.rstrip("/").split("/")
            paper_hash = parts[-1] if parts else ""
            source_id = f"S2-{paper_hash[:16]}" if paper_hash else f"S2-{abs(hash(url)) % 10000000}"

            snippet = str(item.get("content", "") or "").strip()
            score = item.get("score")
            if score is not None:
                try:
                    score = round(float(score), 4)
                except (ValueError, TypeError):
                    score = None

            source = {
                "source_id": source_id,
                "question_id": question_id,
                "query_text": query_text,
                "query_type": "normal",
                "title": title,
                "url": url,
                "normalized_url": normalize_url(url),
                "content": snippet,
                "score": score,
                "discovered_by": [
                    {
                        "question_id": question_id,
                        "query_text": query_text,
                        "query_type": "normal",
                    }
                ],
                "duplicate_count": 0,
                "alternate_urls": [],
                "authors": [],
                "publication_year": None,
                "doi": None,
                "venue": "Semantic Scholar",
                "citation_count": None,
                "abstract": snippet,
                "open_access_url": None,
                "is_open_access": None,
                "provider_name": self.name,
                "provider_query": query_text,
            }
            sources.append(source)

        return sources

    def _search_graph_api(self, query_text, question_id):
        params = {
            "query": query_text,
            "limit": self.limit,
            "fields": (
                "paperId,title,abstract,venue,year,authors,citationCount,"
                "isOpenAccess,openAccessPdf,externalIds,url"
            ),
        }
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, method="GET")
        req.add_header("User-Agent", "research-agent/1.0")
        if self.api_key:
            req.add_header("x-api-key", self.api_key)

        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                if resp.status != 200:
                    return []
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
        except Exception as exc:
            print(f"  [Warning] Semantic Scholar API error: {exc}")
            return []

        raw_items = data.get("data", []) if isinstance(data, dict) else []
        sources = []
        for item in raw_items:
            paper_id = item.get("paperId", "")
            if not paper_id:
                continue
            title = str(item.get("title", "") or "").strip()
            if not title:
                continue

            authors = [
                str(a.get("name", "")).strip()
                for a in item.get("authors", []) or []
                if a.get("name")
            ]
            abstract = str(item.get("abstract", "") or "").strip()
            ext_ids = item.get("externalIds") or {}
            doi = ext_ids.get("DOI")
            paper_url = item.get("url") or f"https://www.semanticscholar.org/paper/{paper_id}"

            source = {
                "source_id": f"S2-{paper_id[:16]}",
                "question_id": question_id,
                "query_text": query_text,
                "query_type": "normal",
                "title": title,
                "url": paper_url,
                "normalized_url": normalize_url(paper_url),
                "content": abstract,
                "score": 0.85,
                "discovered_by": [
                    {
                        "question_id": question_id,
                        "query_text": query_text,
                        "query_type": "normal",
                    }
                ],
                "duplicate_count": 0,
                "alternate_urls": [],
                "authors": authors,
                "publication_year": item.get("year"),
                "doi": f"https://doi.org/{doi}" if doi and not str(doi).startswith("http") else doi,
                "venue": str(item.get("venue", "") or "").strip(),
                "citation_count": item.get("citationCount"),
                "abstract": abstract,
                "open_access_url": (item.get("openAccessPdf") or {}).get("url"),
                "is_open_access": item.get("isOpenAccess"),
                "provider_name": self.name,
                "provider_query": query_text,
            }
            sources.append(source)

        return sources


_providers = {}


def register_provider(name, provider):
    if not isinstance(provider, SourceProvider):
        raise TypeError("Provider must be a SourceProvider instance")
    _providers[name] = provider


def get_provider(name):
    return _providers.get(name)


def search_academic(provider_name, query_text, question_id=None, **kwargs):
    provider = get_provider(provider_name)
    if provider is None:
        raise ValueError(
            f"Unknown source provider: {provider_name}"
        )
    return provider.search(query_text, question_id=question_id, **kwargs)


register_provider("openalex", OpenAlexProvider())
register_provider("semanticscholar", SemanticScholarProvider())


if __name__ == "__main__":
    provider = OpenAlexProvider()
    print("Testing OpenAlex provider...")
    results = provider.search(
        "transformer architecture attention mechanism",
        question_id="TEST",
    )
    print(f"Found {len(results)} results")
    for r in results[:3]:
        print(f"  {r['source_id']}: {r['title']}")
        print(f"    URL: {r['url']}")
        print(f"    Authors: {r.get('authors', [])}")
        print(f"    Year: {r.get('publication_year')}")
        print(f"    Citations: {r.get('citation_count')}")
        print(f"    OA: {r.get('is_open_access')}")
        print(f"    DOI: {r.get('doi')}")
        print(f"    Provider: {r.get('provider_name')}")
        preview = r.get("abstract") or ""
        print(f"    Abstract preview: {preview[:120]}...")
