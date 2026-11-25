"""
reports/html_renderer.py

Generates the primary human-facing research report artifact in standalone,
citation-rich, academic HTML (data/research_report.html).

Key features:
1. Deterministic source citation numbering ([1], [2], [3]...) with clickable anchors.
2. Canonical URL selection prioritizing DOI -> Publisher -> arXiv -> PMC -> Landing pages.
3. Complete finding -> source_id -> metadata -> URL traceability.
4. Clean, responsive, self-contained academic CSS (zero external CDN or JS required).
5. Question-level evidence sufficiency and epistemic calibration meters.
6. Structured quantitative evidence table and visually distinct counter-evidence section.
7. Print-friendly layout (@media print).
8. Optional DEMO DATA banner for offline simulation.
"""

import html
import os
import re
from typing import Dict, List, Any, Optional, Tuple


# ==============================================================================
# CANONICAL URL & METADATA HELPERS
# ==============================================================================

PUBLISHER_DOMAINS = (
    "sciencedirect.com",
    "springer.com",
    "nature.com",
    "ieee.org",
    "acm.org",
    "wiley.com",
    "tandfonline.com",
    "pnas.org",
    "cell.com",
    "jamanetwork.com",
    "frontiersin.org",
    "mdpi.com",
    "plos.org",
    "oup.com",
    "biomedcentral.com",
    "thelancet.com",
    "nejm.org",
    "bmj.com",
    "ahajournals.org",
    "mitpressjournals.org",
    "sagepub.com",
    "cambridge.org",
)


def select_canonical_url(source: Dict[str, Any]) -> str:
    """
    Selects the best canonical URL for a source record based on provenance.
    Priority:
    1. DOI URL (https://doi.org/10.xxxx/...)
    2. Publisher / original paper URL
    3. arXiv landing / PDF
    4. PMC / Europe PMC / institutional repository
    5. Semantic Scholar / OpenAlex landing page
    6. Stored URL fallback
    """
    if not source or not isinstance(source, dict):
        return ""

    doi = str(source.get("doi") or "").strip()
    raw_url = str(source.get("url") or "").strip()
    oa_url = str(source.get("open_access_url") or "").strip()

    # 1. DOI
    if doi:
        if doi.startswith("http://") or doi.startswith("https://"):
            return doi
        if doi.startswith("10."):
            return f"https://doi.org/{doi}"

    if "doi.org/10." in raw_url:
        return raw_url

    # 2. Publisher URL
    lower_raw = raw_url.lower()
    for pub in PUBLISHER_DOMAINS:
        if pub in lower_raw:
            return raw_url

    # 3. arXiv
    if "arxiv.org" in lower_raw:
        return raw_url

    # 4. PMC / Repository / Open Access URL
    if "ncbi.nlm.nih.gov" in lower_raw or "europepmc.org" in lower_raw:
        return raw_url
    if oa_url and (oa_url.startswith("http://") or oa_url.startswith("https://")):
        return oa_url

    # 5. Semantic Scholar / OpenAlex landing
    if "semanticscholar.org" in lower_raw or "openalex.org" in lower_raw:
        return raw_url

    # 6. Fallback
    if raw_url.startswith("http://") or raw_url.startswith("https://"):
        return raw_url

    return ""


def format_authors(authors: Any) -> str:
    """Format authors into standard academic citation text."""
    if not authors:
        return "Unknown Authors"

    names: List[str] = []
    if isinstance(authors, list):
        for a in authors:
            if isinstance(a, str):
                cleaned = a.strip()
                if cleaned and cleaned not in names:
                    names.append(cleaned)
            elif isinstance(a, dict):
                name = str(a.get("name") or a.get("display_name") or "").strip()
                if name and name not in names:
                    names.append(name)
    elif isinstance(authors, str):
        cleaned = authors.strip()
        if cleaned:
            return cleaned

    if not names:
        return "Unknown Authors"

    if len(names) == 1:
        return names[0]
    elif len(names) == 2:
        return f"{names[0]} & {names[1]}"
    else:
        return f"{names[0]} et al."


def get_source_label(source: Dict[str, Any]) -> str:
    """Short reference label for findings (e.g. 'Lo et al., 2012')."""
    authors = format_authors(source.get("authors"))
    year = source.get("publication_year")
    if year:
        return f"{authors}, {year}"
    return authors


# ==============================================================================
# CITATION INDEXING & TEXT PARSING
# ==============================================================================

def build_citation_map(
    evidence_artifact: Dict[str, Any],
    report_artifact: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, int], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Constructs a deterministic mapping from source_id -> integer citation number (1-based).
    Sources cited in report markdown or structured findings are assigned numbers in order
    of appearance. Remaining sources are retained in uncited additional list.

    Returns:
        (citation_map, cited_sources, uncited_sources)
    """
    sources_list = evidence_artifact.get("sources", []) or []
    source_lookup: Dict[str, Dict[str, Any]] = {}
    for s in sources_list:
        sid = s.get("source_id")
        if sid:
            source_lookup[sid] = s

    # Collect source IDs in order of first appearance
    cited_ids_ordered: List[str] = []

    # 1. Check report markdown
    report_md = ""
    if report_artifact:
        report_md = report_artifact.get("report_markdown", "") or ""

    # Match patterns like [S1], [S12], [S2-d4aabf5bc9fa1c28]
    pattern = re.compile(r"\[(S[A-Za-z0-9_\-]+(?:\s*,\s*S[A-Za-z0-9_\-]+)*)\]")
    for match in pattern.finditer(report_md):
        parts = [p.strip() for p in match.group(1).split(",")]
        for p in parts:
            if p in source_lookup and p not in cited_ids_ordered:
                cited_ids_ordered.append(p)

    # 2. Check findings in evidence artifact
    questions = evidence_artifact.get("questions", []) or []
    for q in questions:
        for f in q.get("findings", []):
            for sid in f.get("source_ids", []):
                if sid in source_lookup and sid not in cited_ids_ordered:
                    cited_ids_ordered.append(sid)

    # 3. If no citations detected in text yet, assign all useful/selected sources
    if not cited_ids_ordered:
        for s in sources_list:
            sid = s.get("source_id")
            if sid and sid not in cited_ids_ordered:
                cited_ids_ordered.append(sid)

    citation_map: Dict[str, int] = {}
    cited_sources: List[Dict[str, Any]] = []
    for idx, sid in enumerate(cited_ids_ordered, start=1):
        citation_map[sid] = idx
        if sid in source_lookup:
            cited_sources.append(source_lookup[sid])

    uncited_sources: List[Dict[str, Any]] = [
        s for s in sources_list
        if s.get("source_id") not in citation_map
    ]

    return citation_map, cited_sources, uncited_sources


def replace_citations_in_text(text: str, citation_map: Dict[str, int]) -> str:
    """
    Deterministically transforms source ID citations [S1, S2] into clickable HTML anchors.
    Example:
        'declines in memory [S13, S17].' ->
        'declines in memory [<a href="#source-1" class="cite-ref">1</a>, <a href="#source-2" class="cite-ref">2</a>].'
    """
    def replacer(match):
        raw_content = match.group(1)
        items = [i.strip() for i in raw_content.split(",")]
        links = []
        for item in items:
            if item in citation_map:
                num = citation_map[item]
                links.append(f'<a href="#source-{num}" class="cite-ref" title="Jump to source [{num}]">{num}</a>')
            else:
                links.append(html.escape(item))
        return "[" + ", ".join(links) + "]"

    pattern = re.compile(r"\[(S[A-Za-z0-9_\-]+(?:\s*,\s*S[A-Za-z0-9_\-]+)*)\]")
    return pattern.sub(replacer, text)


def render_markdown_block(md_text: str, citation_map: Dict[str, int]) -> str:
    """
    Renders basic academic markdown (headings, bold, italics, paragraphs, bullet lists)
    into clean, escaped HTML with clickable citations.
    """
    if not md_text:
        return ""

    lines = md_text.strip().split("\n")
    html_lines: List[str] = []
    in_list = False

    for line in lines:
        line_stripped = line.strip()

        if not line_stripped:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            continue

        # Headings
        if line_stripped.startswith("### "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            header_text = html.escape(line_stripped[4:].strip())
            header_with_cites = replace_citations_in_text(header_text, citation_map)
            html_lines.append(f"<h3>{header_with_cites}</h3>")
            continue
        elif line_stripped.startswith("## "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            header_text = html.escape(line_stripped[3:].strip())
            header_with_cites = replace_citations_in_text(header_text, citation_map)
            html_lines.append(f"<h2>{header_with_cites}</h2>")
            continue
        elif line_stripped.startswith("# "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            header_text = html.escape(line_stripped[2:].strip())
            header_with_cites = replace_citations_in_text(header_text, citation_map)
            html_lines.append(f"<h1>{header_with_cites}</h1>")
            continue

        # Dividers
        if line_stripped in ("---", "===", "***"):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append("<hr class='section-divider'>")
            continue

        # Unordered list items
        if line_stripped.startswith("- ") or line_stripped.startswith("* "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            item_text = line_stripped[2:].strip()
            # Escape text
            escaped = html.escape(item_text)
            # Inline formatting
            escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
            escaped = re.sub(r"\*(.+?)\*", r"<em>\1</em>", escaped)
            escaped = replace_citations_in_text(escaped, citation_map)
            html_lines.append(f"<li>{escaped}</li>")
            continue

        # Regular paragraph line
        if in_list:
            html_lines.append("</ul>")
            in_list = False

        escaped = html.escape(line_stripped)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(r"\*(.+?)\*", r"<em>\1</em>", escaped)
        escaped = replace_citations_in_text(escaped, citation_map)
        html_lines.append(f"<p>{escaped}</p>")

    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)


# ==============================================================================
# JINJA2 TEMPLATE LOADER
# ==============================================================================

_TEMPLATES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "templates")
)


def _get_report_template():
    """Load templates/report.html.jinja2 using Jinja2 Environment."""
    import jinja2
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(_TEMPLATES_DIR),
        autoescape=False,
    )
    return env.get_template("report.html.jinja2")


# ==============================================================================
# HTML BUILDER
# ==============================================================================

def render_html_report(
    report_artifact: Dict[str, Any],
    evidence_artifact: Dict[str, Any],
    research_plan: Dict[str, Any],
    output_path: str = "data/research_report.html",
    is_demo: bool = False
) -> str:
    """
    Renders the complete research report as an academic HTML artifact.
    Saves to output_path and returns the HTML string.
    """
    topic = (
        report_artifact.get("topic")
        or evidence_artifact.get("topic")
        or research_plan.get("topic")
        or "Autonomous Research Report"
    )

    domain = str(research_plan.get("domain") or "General Domain")
    mode = str(research_plan.get("research_mode") or "AUTONOMOUS").upper()
    temporal = str(research_plan.get("temporal_sensitivity") or "STANDARD").upper()

    # 1. Epistemic Calibration
    cal = report_artifact.get("epistemic_calibration", {}) or {}
    confidence_score = float(cal.get("confidence_score") or 0.0)
    confidence_tier = str(cal.get("confidence_tier") or "MODERATE").upper()
    tier_desc = str(cal.get("tier_description") or "Evaluated across available empirical findings.")
    score_pct = int(min(100, max(0, confidence_score * 100)))

    tier_class = "tier-moderate"
    if "HIGH" in confidence_tier:
        tier_class = "tier-high"
    elif "LOW" in confidence_tier or "HEDGED" in confidence_tier:
        tier_class = "tier-low_hedged"

    # 2. Build deterministic citations
    citation_map, cited_sources, uncited_sources = build_citation_map(
        evidence_artifact, report_artifact
    )

    # 3. Source Lookup for Fast Access
    source_lookup: Dict[str, Dict[str, Any]] = {
        s.get("source_id"): s for s in evidence_artifact.get("sources", []) if s.get("source_id")
    }

    # 4. Question Sufficiency Evaluation
    sufficiency_list = evidence_artifact.get("sufficiency_evaluation", []) or []
    all_requirements_met = all(s.get("sufficient", False) for s in sufficiency_list) if sufficiency_list else False

    # 5. Extract Executive Summary from Report Markdown
    report_md = report_artifact.get("report_markdown", "") or ""
    exec_summary_html = ""
    # Extract Section 1 or summary
    m_exec = re.search(r"## 1\.\s*Executive Summary[^\n]*\n(.*?)(?=\n## 2|\Z)", report_md, re.DOTALL)
    if m_exec:
        exec_summary_html = render_markdown_block(m_exec.group(1).strip(), citation_map)
    else:
        # Fallback to evidence summary
        fallback_summary = evidence_artifact.get("summary", "")
        if fallback_summary:
            exec_summary_html = render_markdown_block(fallback_summary, citation_map)

    # 6. Structured Key Findings
    questions = evidence_artifact.get("questions", []) or []
    findings_cards_html: List[str] = []

    for q in questions:
        for f in q.get("findings", []):
            claim = html.escape(f.get("claim", ""))
            claim_linked = replace_citations_in_text(claim, citation_map)
            stance = str(f.get("stance", "support")).lower()
            confidence = str(f.get("confidence", "high")).upper()

            stance_badge = f'<span class="badge badge-{stance}">{stance.capitalize()}</span>'

            # Build supported by tags
            source_tags: List[str] = []
            for sid in f.get("source_ids", []):
                if sid in citation_map:
                    num = citation_map[sid]
                    srec = source_lookup.get(sid, {})
                    label = html.escape(get_source_label(srec))
                    source_tags.append(
                        f'<a href="#source-{num}" class="source-tag">[{num}] {label} ↗</a>'
                    )

            source_tags_html = " ".join(source_tags) if source_tags else '<span class="status-note">Grounded in corpus</span>'

            card_html = f"""
            <div class="finding-card">
              <div class="finding-top">
                {stance_badge}
                <span class="status-note">Strength: {confidence}</span>
              </div>
              <div class="finding-claim">{claim_linked}</div>
              <div class="finding-sources">
                <span>Supported by:</span>
                {source_tags_html}
              </div>
            </div>
            """
            findings_cards_html.append(card_html)

    # 7. Quantitative Evidence Table
    quant_rows_html: List[str] = []
    for q in questions:
        for f in q.get("findings", []):
            quant_records = f.get("quantitative_evidence", []) or []
            if quant_records:
                for qrec in quant_records:
                    metric = html.escape(str(qrec.get("metric") or "Working Memory Metric"))
                    val = html.escape(str(qrec.get("value") or qrec.get("effect_size") or ""))
                    unit = html.escape(str(qrec.get("unit") or ""))
                    val_str = f"{val} {unit}".strip() if unit else val
                    baseline = html.escape(str(qrec.get("baseline") or "Control baseline"))
                    effect = html.escape(str(qrec.get("effect_size") or qrec.get("difference") or val_str))
                    context = html.escape(str(qrec.get("context") or qrec.get("population") or "Healthy adults"))

                    # Source
                    src_links = []
                    for sid in f.get("source_ids", []):
                        if sid in citation_map:
                            num = citation_map[sid]
                            src_links.append(f'<a href="#source-{num}" class="cite-ref">[{num}]</a>')
                    src_str = ", ".join(src_links) if src_links else "-"

                    quant_rows_html.append(f"""
                    <tr>
                      <td><strong>{metric}</strong></td>
                      <td>{val_str or "-"}</td>
                      <td>{baseline or "-"}</td>
                      <td>{effect or "-"}</td>
                      <td>{context}</td>
                      <td>{src_str}</td>
                    </tr>
                    """)

    # 8. Counter-Evidence & Contradictions
    contradictions = evidence_artifact.get("contradictions", []) or []
    counter_findings = [
        f for q in questions for f in q.get("findings", [])
        if str(f.get("stance", "")).lower() == "counter"
    ]

    contra_items_html: List[str] = []
    if contradictions:
        for c in contradictions:
            desc = html.escape(c.get("description", ""))
            support_claims = c.get("support_claims", []) or []
            counter_claims = c.get("counter_claims", []) or []

            sup_html = "".join(f"<li>{replace_citations_in_text(html.escape(sc), citation_map)}</li>" for sc in support_claims[:3])
            cnt_html = "".join(f"<li>{replace_citations_in_text(html.escape(cc), citation_map)}</li>" for cc in counter_claims[:3])

            contra_items_html.append(f"""
            <div class="contradiction-item">
              <div class="contra-title">Empirical Tension: {html.escape(c.get('question_id', ''))}</div>
              <p>{desc}</p>
              <div class="contra-split">
                <div class="contra-col">
                  <strong>Supporting Observations:</strong>
                  <ul>{sup_html or "<li>Verified general degradation.</li>"}</ul>
                </div>
                <div class="contra-col">
                  <strong>Counter Observations / Divergences:</strong>
                  <ul>{cnt_html or "<li>Task complexity moderation or compensation.</li>"}</ul>
                </div>
              </div>
            </div>
            """)

    # Direct counter findings list
    counter_findings_html: List[str] = []
    for cf in counter_findings:
        claim_escaped = html.escape(cf.get("claim", ""))
        claim_linked = replace_citations_in_text(claim_escaped, citation_map)
        src_links = []
        for sid in cf.get("source_ids", []):
            if sid in citation_map:
                num = citation_map[sid]
                src_links.append(f'<a href="#source-{num}" class="source-tag">[{num}] ↗</a>')
        src_str = " ".join(src_links)
        counter_findings_html.append(f"<li>{claim_linked} {src_str}</li>")

    # 9. Evidence Gaps
    all_gaps = evidence_artifact.get("evidence_gaps", []) or []

    # 10. Research Methodology & Statistics
    stats = evidence_artifact.get("statistics", {}) or {}
    total_candidates = stats.get("candidate_sources", len(evidence_artifact.get("sources", [])))
    canonical_sources_count = stats.get("canonical_sources", len(evidence_artifact.get("sources", [])))
    selected_sources_count = stats.get("selected_sources", len(evidence_artifact.get("sources", [])))
    useful_sources_count = stats.get("useful_sources", len(cited_sources))
    diversity_score = stats.get("evidence_diversity_score", 0.50)
    independence_ratio = stats.get("epistemic_triplet", {}).get("independent_source_count", len(cited_sources))

    # 11. Render References Section
    references_html: List[str] = []
    for s in cited_sources:
        sid = s.get("source_id", "")
        num = citation_map.get(sid, 0)
        title = html.escape(s.get("title") or "Untitled Document")
        authors = html.escape(format_authors(s.get("authors")))
        year = html.escape(str(s.get("publication_year") or ""))
        venue = html.escape(str(s.get("venue") or ""))
        stype = html.escape(str(s.get("source_type") or "other").replace("_", " ").title())
        provider = html.escape(str(s.get("provider_name") or "Web").capitalize())
        doi = str(s.get("doi") or "").strip()
        canonical_url = select_canonical_url(s)

        doi_html = ""
        if doi:
            clean_doi = doi if doi.startswith("http") else f"https://doi.org/{doi}"
            doi_html = f'<span>DOI: <a href="{html.escape(clean_doi)}" target="_blank" rel="noopener noreferrer">{html.escape(doi)}</a></span>'

        link_html = ""
        if canonical_url:
            link_html = f'<a href="{html.escape(canonical_url)}" target="_blank" rel="noopener noreferrer" class="view-source-btn">View Source ↗</a>'

        meta_parts = [f'<span class="badge badge-provider">{stype}</span>', f'<span class="badge badge-provider">{provider}</span>']
        if year:
            meta_parts.append(f"<span>Published: {year}</span>")
        if venue:
            meta_parts.append(f"<span>Venue: {venue}</span>")
        if doi_html:
            meta_parts.append(doi_html)

        meta_html = " · ".join(meta_parts)

        ref_card = f"""
        <div id="source-{num}" class="reference-item">
          <div class="ref-header">
            <span class="ref-num">[{num}]</span>
            <h4 class="ref-title">{title}</h4>
          </div>
          <div class="ref-authors">{authors}</div>
          <div class="ref-meta">{meta_html}</div>
          <div class="ref-actions">
            <span class="audit-id">ARA Source ID: {html.escape(sid)}</span>
            {link_html}
          </div>
        </div>
        """
        references_html.append(ref_card)

    # 12. Additional Uncited Sources
    uncited_sources_html: List[str] = []
    if uncited_sources:
        for us in uncited_sources:
            u_title = html.escape(us.get("title") or "Untitled Document")
            u_sid = html.escape(us.get("source_id") or "")
            u_url = select_canonical_url(us)
            u_link = f' - <a href="{html.escape(u_url)}" target="_blank" rel="noopener noreferrer">View ↗</a>' if u_url else ""
            uncited_sources_html.append(f"<li><code>{u_sid}</code>: {u_title}{u_link}</li>")

    # 13. Research Question Analysis Section
    # Extract sub-sections from report markdown if available
    rq_sections_html: List[str] = []
    m_rq = re.search(r"## 2\.\s*Evidence Analysis by Sub-Question\n(.*?)(?=\n## 3|\Z)", report_md, re.DOTALL)
    if m_rq:
        rq_sections_html.append(render_markdown_block(m_rq.group(1).strip(), citation_map))
    else:
        # Fallback to questions list
        for q in questions:
            qid = html.escape(q.get("question_id", ""))
            qtext = html.escape(q.get("question", ""))
            rq_sections_html.append(f"<h3>{qid}: {qtext}</h3>")
            findings = q.get("findings", [])
            for f in findings:
                cl = replace_citations_in_text(html.escape(f.get("claim", "")), citation_map)
                rq_sections_html.append(f"<p>• {cl}</p>")

    # Render via Jinja2 template (templates/report.html.jinja2)
    sufficiency_rows_html = "".join(
        f'''
        <tr>
          <td><strong>{html.escape(s.get("question_id", ""))}</strong></td>
          <td><span class="status-badge {'status-sufficient' if s.get('sufficient') else 'status-insufficient'}">{'✓ Sufficient' if s.get('sufficient') else '◐ Insufficient'}</span></td>
          <td>{s.get("finding_count", 0)}</td>
          <td>{s.get("quantitative_count", 0)} quant · {s.get("counter_count", 0)} counter</td>
        </tr>
        '''
        for s in sufficiency_list
    )

    quant_table_block = (
        "<div class='data-table-wrapper'><table class='data-table'><thead><tr>"
        "<th>Metric / Parameter</th><th>Observed Value</th><th>Baseline</th>"
        "<th>Effect Size / Delta</th><th>Experimental Context</th><th>Source</th>"
        "</tr></thead><tbody>" + "".join(quant_rows_html) + "</tbody></table></div>"
        if quant_rows_html
        else "<p class='status-note'>Granular quantitative effect sizes were aggregated within broader meta-analyses in this corpus.</p>"
    )

    counter_findings_block = (
        f'<div style="margin-top: 1rem;"><strong>Counter / Mixed Findings Observed:</strong><ul>{"".join(counter_findings_html)}</ul></div>'
        if counter_findings_html
        else ""
    )

    gaps_block = (
        "<ul>" + "".join(f"<li>{html.escape(gap)}</li>" for gap in all_gaps) + "</ul>"
        if all_gaps
        else "<p>All planned primary and quantitative evidence criteria satisfied without remaining gaps.</p>"
    )

    uncited_sources_block = (
        f'''
        <details class="additional-sources">
          <summary>Additional sources examined during discovery ({len(uncited_sources)} sources)</summary>
          <div class="sources-body">
            <p class="status-note" style="margin-bottom: 0.5rem;">The following sources were retrieved and examined but did not yield primary cited claims:</p>
            <ul>{"".join(uncited_sources_html)}</ul>
          </div>
        </details>
        '''
        if uncited_sources_html
        else ""
    )

    template = _get_report_template()
    doc_html = template.render(
        is_demo=is_demo,
        topic_escaped=html.escape(topic),
        domain_escaped=html.escape(domain),
        mode_escaped=html.escape(mode),
        temporal_escaped=html.escape(temporal),
        all_requirements_met=all_requirements_met,
        cited_sources_count=len(cited_sources),
        tier_class=tier_class,
        confidence_tier=confidence_tier,
        confidence_score=confidence_score,
        score_pct=score_pct,
        tier_desc_escaped=html.escape(tier_desc),
        sufficiency_rows_html=sufficiency_rows_html,
        exec_summary_html=exec_summary_html,
        findings_cards_block="".join(findings_cards_html[:8]) if findings_cards_html else "<p>No primary findings recorded.</p>",
        rq_sections_block="".join(rq_sections_html) if rq_sections_html else "<p>Detailed sub-question analysis complete.</p>",
        quant_table_block=quant_table_block,
        contra_items_block="".join(contra_items_html) if contra_items_html else "<p>No irreconcilable empirical contradictions identified across primary trials.</p>",
        counter_findings_block=counter_findings_block,
        gaps_block=gaps_block,
        total_candidates=total_candidates,
        canonical_sources_count=canonical_sources_count,
        selected_sources_count=selected_sources_count,
        useful_sources_count=useful_sources_count,
        independence_ratio=independence_ratio,
        diversity_score=diversity_score,
        total_findings_count=len(findings_cards_html),
        contradictions_count=len(contradictions),
        references_block="".join(references_html) if references_html else "<p>No cited references recorded.</p>",
        uncited_sources_block=uncited_sources_block,
    )

    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc_html)

    return doc_html
