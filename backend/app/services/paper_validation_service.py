import re
import os
from collections import Counter


SECTION_PATTERNS = [
    ("introduction", r"\b(?:I\.?\s+)?Introduction\b"),
    ("related_work", r"\b(?:II\.?\s+)?(?:Related\s+Work|Background|Preliminaries|Literature\s+Review)\b"),
    ("methodology", r"\b(?:III\.?\s+)?(?:Method(?:ology|s)?|Approach|Framework|Model(?:s)?)\b"),
    ("experiments", r"\b(?:IV\.?\s+)?(?:Experiments?|Evaluation|Results?|Empirical|Experimental\s+Setup)\b"),
    ("discussion", r"\b(?:V\.?\s+)?Discussion\b"),
    ("conclusion", r"\b(?:VI\.?\s+)?(?:Conclusions?|Summary|Final\s+Remarks)\b"),
]

REFERENCE_SECTION_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:References|Bibliography|WORKS?\s+CITED)\s*(?:\n|$)",
    re.IGNORECASE,
)

REFERENCE_ENTRY_PATTERNS = [
    re.compile(r"\[\d+\]", re.IGNORECASE),
    re.compile(r"^\d+\.\s+", re.MULTILINE),
    re.compile(r"^\([a-z]\)", re.MULTILINE),
]

CITATION_PATTERNS = [
    re.compile(r"\[\d+(?:[-–,]\d+)*\]", re.IGNORECASE),
    re.compile(r"\((?:[A-Z][a-z]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][a-z]+)?(?:,?\s*\d{4})+|(?:[A-Z][a-z]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][a-z]+)?,?\s*\d{4}(?:;\s*[A-Z][a-z]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][a-z]+)?,?\s*\d{4})*))\)"),
]

FIGURE_TABLE_PATTERNS = [
    re.compile(r"(?:Fig(?:ure|\.?)|Diagram|Graph|Plot)\s*\d+", re.IGNORECASE),
    re.compile(r"(?:Table|Tab\.?)\s*\d+", re.IGNORECASE),
]


def _word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def _paragraphs(text: str) -> list[str]:
    if not text:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]


def _check_pdf_integrity(pdf_path: str | None) -> dict:
    result = {
        "pdf_accessible": False,
        "pdf_readable": False,
        "pdf_encrypted": False,
        "pdf_corrupted": False,
        "pdf_page_count": None,
    }

    if not pdf_path:
        result["pdf_accessible"] = False
        return result

    if not os.path.isfile(pdf_path):
        result["pdf_accessible"] = False
        return result

    result["pdf_accessible"] = True

    try:
        import fitz
        doc = fitz.open(pdf_path)
        try:
            result["pdf_page_count"] = doc.page_count
            result["pdf_encrypted"] = doc.is_encrypted
            result["pdf_readable"] = doc.page_count > 0 and not doc.is_encrypted
        finally:
            doc.close()
    except Exception:
        result["pdf_corrupted"] = True
        result["pdf_readable"] = False

    return result


def _check_metadata(paper) -> dict:
    return {
        "title_present": bool(paper.title and str(paper.title).strip()),
        "authors_present": bool(paper.authors and str(paper.authors).strip()),
        "abstract_present": bool(paper.abstract and str(paper.abstract).strip()),
        "doi_present": bool(paper.doi and str(paper.doi).strip()),
        "source_present": bool(paper.source and str(paper.source).strip()),
        "published_date_present": paper.published_date is not None,
        "categories_present": bool(paper.categories and str(paper.categories).strip()),
    }


def _check_sections(full_text: str) -> dict:
    if not full_text:
        return {"sections_detected": [], "section_count": 0}

    detected = []
    for name, pattern in SECTION_PATTERNS:
        if re.search(pattern, full_text, re.IGNORECASE):
            detected.append(name)

    return {
        "sections_detected": detected,
        "section_count": len(detected),
    }


def _check_references(full_text: str) -> dict:
    if not full_text:
        return {
            "has_references_section": False,
            "reference_count": 0,
            "reference_counting_method": "not_available",
        }

    has_refs = bool(REFERENCE_SECTION_PATTERN.search(full_text))

    if not has_refs:
        return {
            "has_references_section": False,
            "reference_count": 0,
            "reference_counting_method": "not_available",
        }

    ref_section_match = REFERENCE_SECTION_PATTERN.search(full_text)
    if ref_section_match:
        ref_text = full_text[ref_section_match.end():]
    else:
        ref_text = full_text

    bracket_refs = re.findall(r"\[\d+\]", ref_text)
    if bracket_refs:
        nums = [int(re.sub(r"[^\d]", "", b)) for b in bracket_refs if re.sub(r"[^\d]", "", b)]
        return {
            "has_references_section": True,
            "reference_count": max(nums) if nums else len(bracket_refs),
            "reference_counting_method": "bracket_numbering",
        }

    lines = [l.strip() for l in ref_text.split("\n") if l.strip()]

    entry_count = 0
    for l in lines:
        has_year = bool(re.search(r"\b(?:19|20)\d{2}\b", l))
        if not has_year:
            continue
        starts_with_author = bool(re.match(r"^([A-Z][.\s]|[A-Z][a-z]|\()", l))
        has_journal_or_conference = bool(re.search(
            r"(?:journal|proceedings|conf\.|workshop|arxiv|preprint|press|springer|ieee|acm|aaai|nips|icml|iclr|cvpr|eccv|emnlp|acl|naacl|iccv|wscl|ssrn)",
            l,
            re.IGNORECASE,
        ))
        has_et_al_or_multiple_authors = bool(re.search(
            r"(?:et\s+al|,\s*[A-Z]|and\s+[A-Z])",
            l,
        ))
        if starts_with_author and (has_journal_or_conference or has_et_al_or_multiple_authors):
            entry_count += 1

    if entry_count > 0:
        return {
            "has_references_section": True,
            "reference_count": entry_count,
            "reference_counting_method": "heuristic_entry_count",
        }

    numbered = re.findall(r"^\d+\.\s+", ref_text, re.MULTILINE)
    if numbered:
        return {
            "has_references_section": True,
            "reference_count": len(numbered),
            "reference_counting_method": "numbered_entries",
        }

    return {
        "has_references_section": True,
        "reference_count": 0,
        "reference_counting_method": "unable_to_count",
    }


def _check_citations(full_text: str) -> dict:
    if not full_text:
        return {
            "in_text_citation_count": 0,
            "citation_formats_found": [],
            "citations_detected": False,
        }

    found_formats = []
    all_citations = []

    bracket_cites = re.findall(r"\[\d+(?:[-–,]\d+)*\]", full_text)
    if bracket_cites:
        found_formats.append("bracket_number")
        all_citations.extend(bracket_cites)

    paren_name_cites = re.findall(
        r"\([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and|&)\s*[A-Z]?[a-z]*)?,?\s*\d{4}(?:;\s*[A-Z][a-z]+(?:\s+(?:et\s+al\.?|and|&)\s*[A-Z]?[a-z]*)?,?\s*\d{4})*\)",
        full_text,
    )
    if paren_name_cites:
        found_formats.append("author_year")
        all_citations.extend(paren_name_cites)

    return {
        "in_text_citation_count": len(all_citations),
        "citation_formats_found": found_formats,
        "citations_detected": len(all_citations) > 0,
    }


def _check_citation_consistency(
    ref_data: dict,
    cite_data: dict,
) -> dict:
    has_refs = ref_data.get("has_references_section", False)
    has_cites = cite_data.get("citations_detected", False)
    ref_count = ref_data.get("reference_count", 0)
    cite_count = cite_data.get("in_text_citation_count", 0)
    method = ref_data.get("reference_counting_method", "not_available")

    if not has_refs and not has_cites:
        return {
            "consistent": None,
            "detail": "No references section or in-text citations detected",
        }

    if not has_refs:
        return {
            "consistent": None,
            "detail": "In-text citations found but no references section detected",
        }

    if not has_cites:
        return {
            "consistent": None,
            "detail": "References section found but no in-text citations detected",
        }

    if method in ("not_available", "unable_to_count"):
        return {
            "consistent": None,
            "detail": f"Cannot reliably count references ({method})",
        }

    if ref_count > 0 and cite_count > 0:
        ratio = cite_count / ref_count
        consistent = 0.3 <= ratio <= 10.0
        return {
            "consistent": consistent,
            "detail": f"{cite_count} citations vs {ref_count} references (ratio: {ratio:.1f})",
        }

    return {
        "consistent": None,
        "detail": "Unable to compare citation and reference counts",
    }


def _check_figures_tables(full_text: str) -> dict:
    if not full_text:
        return {
            "figures_detected": 0,
            "tables_detected": 0,
            "total_figures_tables": 0,
        }

    fig_refs = []
    for match in re.finditer(r"(?:Fig(?:ure|\.?)|Diagram|Graph|Plot)\s*(\d+)", full_text, re.IGNORECASE):
        fig_refs.append(match.group(1))

    tab_refs = []
    for match in re.finditer(r"(?:Table|Tab\.?)\s*(\d+)", full_text, re.IGNORECASE):
        tab_refs.append(match.group(1))

    return {
        "figures_detected": len(set(fig_refs)),
        "tables_detected": len(set(tab_refs)),
        "total_figures_tables": len(set(fig_refs)) + len(set(tab_refs)),
    }


def _check_empty_short_pages(chunks: list) -> dict:
    if not chunks:
        return {
            "empty_pages": [],
            "short_pages": [],
            "empty_page_count": 0,
            "short_page_count": 0,
        }

    page_texts: dict[int, str] = {}
    for chunk in chunks:
        pn = chunk.page_number
        if pn is not None:
            if pn not in page_texts:
                page_texts[pn] = ""
            page_texts[pn] += " " + (chunk.text or "")

    if not page_texts:
        return {
            "empty_pages": [],
            "short_pages": [],
            "empty_page_count": 0,
            "short_page_count": 0,
        }

    all_pages = sorted(page_texts.keys())
    max_page = max(all_pages) if all_pages else 0

    empty = []
    short = []
    SHORT_THRESHOLD = 50

    for page_num in range(1, max_page + 1):
        text = page_texts.get(page_num, "")
        wc = _word_count(text)
        if wc == 0:
            empty.append(page_num)
        elif wc < SHORT_THRESHOLD:
            short.append(page_num)

    return {
        "empty_pages": empty,
        "short_pages": short,
        "empty_page_count": len(empty),
        "short_page_count": len(short),
    }


def _check_duplicate_text(full_text: str) -> dict:
    if not full_text:
        return {
            "duplicate_paragraphs": 0,
            "duplicate_paragraph_ratio": 0.0,
            "has_duplicates": False,
        }

    paras = _paragraphs(full_text)
    non_empty = [p for p in paras if _word_count(p) >= 20]

    if len(non_empty) < 2:
        return {
            "duplicate_paragraphs": 0,
            "duplicate_paragraph_ratio": 0.0,
            "has_duplicates": False,
        }

    normalized = []
    for p in non_empty:
        n = re.sub(r"\s+", " ", p.lower().strip())
        n = re.sub(r"[^\w\s]", "", n)
        normalized.append(n)

    counts = Counter(normalized)
    dupes = sum(c - 1 for c in counts.values() if c > 1)

    return {
        "duplicate_paragraphs": dupes,
        "duplicate_paragraph_ratio": round(dupes / len(non_empty), 4) if non_empty else 0.0,
        "has_duplicates": dupes > 0,
    }


def _check_structural_issues(full_text: str) -> list[str]:
    issues = []
    if not full_text:
        return issues

    if re.search(r"\n{5,}", full_text):
        issues.append("Excessive blank lines detected (5+ consecutive)")

    long_paras = [
        p for p in _paragraphs(full_text)
        if _word_count(p) > 800
    ]
    if long_paras:
        issues.append(f"{len(long_paras)} paragraph(s) exceed 800 words")

    if re.search(r"[\x00-\x08\x0e-\x1f]", full_text):
        issues.append("Non-printable control characters found in text")

    words = full_text.split()
    if words:
        caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / len(words)
        if caps_ratio > 0.3:
            issues.append("Unusually high ratio of ALL-CAPS words")

    return issues


def validate_paper_full(paper, chunks: list | None = None) -> dict:
    full_text = paper.full_text or ""
    wc = _word_count(full_text)

    pdf_info = _check_pdf_integrity(paper.pdf_path)
    metadata = _check_metadata(paper)
    sections = _check_sections(full_text)
    ref_data = _check_references(full_text)
    cite_data = _check_citations(full_text)
    consistency = _check_citation_consistency(ref_data, cite_data)
    fig_tab = _check_figures_tables(full_text)
    structural = _check_structural_issues(full_text)

    needs_ocr = False
    if pdf_info["pdf_page_count"] and pdf_info["pdf_page_count"] > 0:
        avg_chars = len(full_text) / pdf_info["pdf_page_count"]
        needs_ocr = avg_chars < 100

    empty_short = _check_empty_short_pages(chunks or [])
    dupes = _check_duplicate_text(full_text)

    checks = []

    checks.append({
        "id": "pdf_readable",
        "name": "PDF Readability",
        "status": "pass" if pdf_info["pdf_readable"] else ("fail" if pdf_info["pdf_accessible"] else "not_applicable"),
        "detail": (
            "PDF opened successfully" if pdf_info["pdf_readable"]
            else ("PDF is corrupted or encrypted" if pdf_info["pdf_accessible"]
                  else "No PDF file found on disk")
        ),
        "value": pdf_info["pdf_readable"],
    })

    checks.append({
        "id": "text_extracted",
        "name": "Text Extraction",
        "status": "pass" if full_text.strip() else "fail",
        "detail": f"Extracted {wc:,} words" if full_text.strip() else "No text content available",
        "value": bool(full_text.strip()),
    })

    checks.append({
        "id": "ocr_required",
        "name": "OCR Requirement",
        "status": "not_applicable" if not pdf_info["pdf_page_count"] else ("warning" if needs_ocr else "pass"),
        "detail": (
            "Not measured" if not pdf_info["pdf_page_count"]
            else ("OCR may be needed" if needs_ocr else "Text extraction sufficient")
        ),
        "value": needs_ocr if pdf_info["pdf_page_count"] else None,
    })

    meta_pass = sum(1 for v in metadata.values() if v)
    meta_total = len(metadata)
    checks.append({
        "id": "metadata_completeness",
        "name": "Metadata Completeness",
        "status": "pass" if meta_pass == meta_total else ("warning" if meta_pass >= 4 else "fail"),
        "detail": f"{meta_pass}/{meta_total} metadata fields populated",
        "value": {"present": meta_pass, "total": meta_total, **metadata},
    })

    checks.append({
        "id": "title_present",
        "name": "Title Presence",
        "status": "pass" if metadata["title_present"] else "fail",
        "detail": "Title found" if metadata["title_present"] else "Title missing",
        "value": metadata["title_present"],
    })

    checks.append({
        "id": "authors_present",
        "name": "Author Presence",
        "status": "pass" if metadata["authors_present"] else "fail",
        "detail": "Authors found" if metadata["authors_present"] else "Authors missing",
        "value": metadata["authors_present"],
    })

    checks.append({
        "id": "abstract_present",
        "name": "Abstract Presence",
        "status": "pass" if metadata["abstract_present"] else "fail",
        "detail": "Abstract found" if metadata["abstract_present"] else "Abstract missing",
        "value": metadata["abstract_present"],
    })

    checks.append({
        "id": "research_sections",
        "name": "Research Sections",
        "status": "pass" if sections["section_count"] >= 3 else ("warning" if sections["section_count"] >= 1 else "fail"),
        "detail": (
            f"{sections['section_count']} section(s) detected: {', '.join(sections['sections_detected'])}"
            if sections["sections_detected"]
            else "No standard research sections detected"
        ),
        "value": sections,
    })

    checks.append({
        "id": "references_present",
        "name": "References Presence & Count",
        "status": "pass" if ref_data["has_references_section"] else "warning",
        "detail": (
            f"References section found, ~{ref_data['reference_count']} entries ({ref_data['reference_counting_method']})"
            if ref_data["has_references_section"]
            else "No references section detected"
        ),
        "value": ref_data,
    })

    checks.append({
        "id": "in_text_citations",
        "name": "In-text Citation Detection",
        "status": "pass" if cite_data["citations_detected"] else "warning",
        "detail": (
            f"{cite_data['in_text_citation_count']} citation(s) detected ({', '.join(cite_data['citation_formats_found'])})"
            if cite_data["citations_detected"]
            else "No in-text citations detected"
        ),
        "value": cite_data,
    })

    checks.append({
        "id": "citation_consistency",
        "name": "Citation/Reference Consistency",
        "status": (
            "pass" if consistency["consistent"] is True
            else ("warning" if consistency["consistent"] is False
                  else "not_measured")
        ),
        "detail": consistency["detail"],
        "value": consistency,
    })

    checks.append({
        "id": "figures_tables",
        "name": "Figures & Tables Detection",
        "status": "pass" if fig_tab["total_figures_tables"] > 0 else "warning",
        "detail": (
            f"{fig_tab['figures_detected']} figure(s), {fig_tab['tables_detected']} table(s) referenced"
            if fig_tab["total_figures_tables"] > 0
            else "No figure/table references detected"
        ),
        "value": fig_tab,
    })

    checks.append({
        "id": "empty_short_pages",
        "name": "Empty/Short Pages",
        "status": (
            "pass" if empty_short["empty_page_count"] == 0 and empty_short["short_page_count"] == 0
            else "warning"
        ),
        "detail": (
            f"{empty_short['empty_page_count']} empty, {empty_short['short_page_count']} short page(s) (via chunks)"
            if (empty_short["empty_page_count"] > 0 or empty_short["short_page_count"] > 0)
            else "No empty or very short pages detected (via chunks)"
        ),
        "value": empty_short,
    })

    checks.append({
        "id": "duplicate_text",
        "name": "Duplicate Text Detection",
        "status": "pass" if not dupes["has_duplicates"] else "warning",
        "detail": (
            f"{dupes['duplicate_paragraphs']} duplicate paragraph(s) ({dupes['duplicate_paragraph_ratio']:.1%} ratio)"
            if dupes["has_duplicates"]
            else "No duplicate paragraphs detected"
        ),
        "value": dupes,
    })

    checks.append({
        "id": "structural_issues",
        "name": "Structural/Formatting Issues",
        "status": "pass" if not structural else "warning",
        "detail": (
            f"{len(structural)} issue(s): {'; '.join(structural)}"
            if structural
            else "No structural issues detected"
        ),
        "value": structural,
    })

    status_counts = Counter(c["status"] for c in checks)
    fails = status_counts.get("fail", 0)
    warnings = status_counts.get("warning", 0)

    if fails > 0:
        overall = "fail"
    elif warnings > 0:
        overall = "warning"
    else:
        overall = "pass"

    issues_list = [c["detail"] for c in checks if c["status"] == "fail"]
    warnings_list = [c["detail"] for c in checks if c["status"] == "warning"]

    metadata_fields = {
        "title": bool(paper.title and str(paper.title).strip()),
        "authors": bool(paper.authors and str(paper.authors).strip()),
        "abstract": bool(paper.abstract and str(paper.abstract).strip()),
        "doi": bool(paper.doi and str(paper.doi).strip()),
        "source": bool(paper.source and str(paper.source).strip()),
        "published_date": paper.published_date is not None,
        "categories": bool(paper.categories and str(paper.categories).strip()),
    }

    return {
        "status": "success",
        "overall_status": overall,
        "paper_id": paper.id,
        "paper_title": paper.title,
        "checks": checks,
        "metrics": {
            "word_count": wc,
            "page_count": pdf_info["pdf_page_count"],
            "chunk_count": len(chunks) if chunks else 0,
            "section_count": sections["section_count"],
            "reference_count": ref_data["reference_count"],
            "citation_count": cite_data["in_text_citation_count"],
            "figure_count": fig_tab["figures_detected"],
            "table_count": fig_tab["tables_detected"],
            "duplicate_paragraph_count": dupes["duplicate_paragraphs"],
        },
        "metadata_fields": metadata_fields,
        "issues": issues_list,
        "warnings": warnings_list,
    }
