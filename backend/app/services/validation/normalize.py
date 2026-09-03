from datetime import datetime, timezone


# ---------------------------------------------------------------
# arXiv category code → human-readable research domain
# ---------------------------------------------------------------
ARXIV_CATEGORY_MAP: dict[str, str] = {
    # Computer Science
    "cs.AI": "Artificial Intelligence",
    "cs.CV": "Computer Vision",
    "cs.CL": "Natural Language Processing",
    "cs.LG": "Machine Learning",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.RO": "Robotics",
    "cs.MA": "Multi-Agent Systems",
    "cs.IR": "Information Retrieval",
    "cs.CR": "Cryptography and Security",
    "cs.SD": "Sound",
    "cs.MM": "Multimedia",
    "cs.IT": "Information Theory",
    "cs.DS": "Data Structures and Algorithms",
    "cs.DC": "Distributed Computing",
    "cs.HC": "Human-Computer Interaction",
    "cs.CY": "Computers and Society",
    "cs.NI": "Networking",
    "cs.CG": "Computational Geometry",
    "cs.PL": "Programming Languages",
    "cs.SE": "Software Engineering",
    "cs.DB": "Databases",
    "cs.GL": "General Literature",
    "cs.OH": "Other Computer Science",
    # Electrical Engineering and Systems Science
    "eess.AS": "Audio and Speech Processing",
    "eess.IV": "Image and Video Processing",
    "eess.SP": "Signal Processing",
    "eess.SY": "Systems and Control",
    # Statistics
    "stat.ML": "Machine Learning",
    "stat.AP": "Applied Statistics",
    "stat.TH": "Statistics Theory",
    "stat.CO": "Computation",
    "stat.ME": "Methodology",
    "stat.OT": "Other Statistics",
}

# Pattern prefixes that represent invalid / meaningless categories
_INVALID_EXACT = {"string", "none", "null", "undefined", "n/a", "na", "unknown"}


def _parse_datetime(value):
    """Coerce source date strings into naive UTC datetimes.

    The papers table uses naive DateTime columns and asyncpg does not
    accept strings or tz-aware values for them.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)

    return parsed


def normalize_paper(
    paper: dict,
    source: str,
) -> dict:

    # Handle:
    # {"paper": {...}}
    # as well as:
    # {...}
    if "paper" in paper and isinstance(paper["paper"], dict):
        paper = paper["paper"]

    # -------------------------
    # TITLE
    # -------------------------
    title = (
        paper.get("title")
        or paper.get("display_name")
        or paper.get("name")
    )

    if not title or not str(title).strip():
        raise ValueError("Paper title is missing")

    title = str(title).strip()

    # -------------------------
    # AUTHORS
    # -------------------------
    authors = paper.get("authors")

    if not authors:
        authors = []

        for authorship in paper.get("authorships", []):
            author = authorship.get("author", {})

            name = author.get("display_name")

            if name:
                authors.append(name)

    # Make sure authors is always a list
    if not isinstance(authors, list):
        authors = [str(authors)]

    # Remove empty author names
    authors = [
        str(author).strip()
        for author in authors
        if author and str(author).strip()
    ]

    # -------------------------
    # ABSTRACT
    # -------------------------
    abstract = (
        paper.get("abstract")
        or paper.get("description")
        or ""
    )

    if not isinstance(abstract, str):
        abstract = str(abstract)

    # -------------------------
    # DOI
    # -------------------------
    doi = paper.get("doi")

    # -------------------------
    # URL
    # -------------------------
    primary_location = paper.get("primary_location") or {}

    url = (
        paper.get("url")
        or paper.get("landing_page_url")
        or primary_location.get("landing_page_url")
    )

    # -------------------------
    # PDF URL
    # -------------------------
    pdf_url = paper.get("pdf_url")

    if not pdf_url:
        pdf_url = primary_location.get("pdf_url")

    # -------------------------
    # SOURCE ID
    # -------------------------
    source_id = paper.get("source_id")

    if not source_id:
        source_id = paper.get("id")

    # -------------------------
    # ARXIV ID
    # -------------------------
    arxiv_id = paper.get("arxiv_id")

    # -------------------------
    # DATES
    # -------------------------
    published_date = _parse_datetime(
        paper.get("published_date")
    )
    updated_date = _parse_datetime(
        paper.get("updated_date")
    )

    # -------------------------
    # CATEGORIES
    # -------------------------
    raw_categories = paper.get("categories") or []

    # If categories arrived as a single comma-separated string
    # (e.g. from a previous ingestion step), split them first.
    if isinstance(raw_categories, str):
        raw_categories = [
            c.strip() for c in raw_categories.split(",") if c.strip()
        ]

    if not isinstance(raw_categories, list):
        raw_categories = [str(raw_categories)]

    # Normalize each category:
    # 1. Strip whitespace
    # 2. Map arXiv codes → human-readable names
    # 3. Discard invalid / meaningless values
    categories: list[str] = []
    seen: set[str] = set()

    for raw_cat in raw_categories:
        cat = str(raw_cat).strip()
        if not cat:
            continue

        # Discard the literal word "string" and other sentinel values
        if cat.lower() in _INVALID_EXACT:
            continue

        # Try exact arXiv code lookup
        mapped = ARXIV_CATEGORY_MAP.get(cat)

        if mapped:
            key = mapped
        else:
            key = cat

        # Deduplicate (case-insensitive)
        if key.lower() not in seen:
            seen.add(key.lower())
            categories.append(mapped if mapped else cat)

    # -------------------------
    # NORMALIZED RESULT
    # -------------------------
    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "doi": doi,
        "url": url,
        "pdf_url": pdf_url,
        "source": source,

        "source_id": source_id,
        "arxiv_id": arxiv_id,
        "published_date": published_date,
        "updated_date": updated_date,
        "categories": categories,
    }