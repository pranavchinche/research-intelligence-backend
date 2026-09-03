import re
import fitz


OCR_MIN_CHARS_PER_PAGE = 100


# Lines that are very unlikely to be author names.
_AFFILIATION_MARKERS = (
    "university",
    "institute",
    "department",
    "laboratory",
    "lab",
    "college",
    "school",
    "faculty",
    "center",
    "centre",
    "hospital",
    "cern",
    "switzerland",
    "germany",
    "france",
    "italy",
    "uk",
    "usa",
    "united states",
    "email",
    "@",
)


def _clean_metadata_value(value: str | None) -> str | None:
    """Clean a PDF metadata string and return None when unusable."""
    if not value:
        return None

    value = str(value).replace("\x00", " ").strip()
    value = re.sub(r"\s+", " ", value)

    if not value:
        return None

    return value


def clean_pdf_text(text: str) -> str:
    """
    Clean obvious PDF extraction artifacts while preserving
    meaningful research text.
    """

    if not text:
        return ""

    text = re.sub(
        r"<(?:EOS|PAD|BOS|UNK)>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n\s*\n+", "\n\n", text)

    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    return text.strip()


def needs_ocr(full_text: str, page_count: int) -> bool:
    if page_count == 0:
        return True

    text_length = len(full_text.strip())
    average_chars_per_page = text_length / page_count

    return average_chars_per_page < OCR_MIN_CHARS_PER_PAGE


def _looks_like_author_line(line: str) -> bool:
    """
    Conservative heuristic for author lines on the first page.

    This is intentionally not an aggressive name extractor.
    It attempts to avoid treating affiliations, dates, section headings,
    equations, and abstract text as authors.
    """

    if not line:
        return False

    line = re.sub(r"\s+", " ", line).strip()

    if len(line) < 3 or len(line) > 180:
        return False

    lower = line.lower()

    # Obvious non-author lines.
    if lower in {
        "abstract",
        "introduction",
        "contents",
        "keywords",
        "preprint",
        "manuscript",
    }:
        return False

    if re.match(
        r"^(arxiv|doi|https?://|www\.)",
        lower,
    ):
        return False

    # Dates.
    if re.search(
        r"\b(?:january|february|march|april|may|june|july|"
        r"august|september|october|november|december)\b",
        lower,
    ):
        return False

    if re.search(r"\b\d{1,2}\s+\w+\s+\d{4}\b", lower):
        return False

    # Affiliations/contact information.
    if any(marker in lower for marker in _AFFILIATION_MARKERS):
        return False

    # Lines containing too many digits are usually metadata/equations.
    digit_count = sum(character.isdigit() for character in line)

    if digit_count > max(2, len(line) // 8):
        return False

    # Reject very long prose sentences.
    if len(re.findall(r"\s", line)) > 20:
        return False

    # Author names normally contain alphabetic characters.
    if not re.search(r"[A-Za-z]", line):
        return False

    return True


def _extract_authors_from_first_page(
    first_page_text: str,
    title: str | None,
) -> str | None:
    """
    Extract a conservative author block from the first page when
    embedded PDF metadata does not contain authors.

    Strategy:
    - Start searching after the detected title when possible.
    - Inspect only the early part of the first page.
    - Collect one or more plausible author lines.
    - Stop once affiliation/abstract/section content begins.
    """

    if not first_page_text:
        return None

    raw_lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in first_page_text.splitlines()
    ]

    lines = [
        line
        for line in raw_lines
        if line
    ]

    if not lines:
        return None

    title_clean = _clean_metadata_value(title)

    start_index = 0

    if title_clean:
        normalized_title = re.sub(
            r"\s+",
            " ",
            title_clean,
        ).strip().lower()

        for index, line in enumerate(lines[:30]):
            normalized_line = re.sub(
                r"\s+",
                " ",
                line,
            ).strip().lower()

            if (
                normalized_line == normalized_title
                or normalized_title in normalized_line
                or normalized_line in normalized_title
            ):
                start_index = index + 1
                break

    # Only inspect a small title/author region.
    candidate_lines = lines[
        start_index:start_index + 12
    ]

    authors: list[str] = []

    for line in candidate_lines:

        lower = line.lower()

        # Once the abstract/introduction starts, stop.
        if lower in {
            "abstract",
            "introduction",
            "i. introduction",
            "1 introduction",
            "keywords",
        }:
            break

        # Affiliation lines usually indicate the end of the author block.
        if any(
            marker in lower
            for marker in _AFFILIATION_MARKERS
        ):
            continue

        if _looks_like_author_line(line):
            authors.append(line)

        # Most papers have only a small author block.
        if len(authors) >= 4:
            break

    if not authors:
        return None

    # Remove obvious duplicates while preserving order.
    unique_authors = []

    for author in authors:
        if author not in unique_authors:
            unique_authors.append(author)

    if not unique_authors:
        return None

    # Normalize "A and B" / "A, B and C" into a single clean string.
    return ", ".join(unique_authors)


def extract_pdf_text(pdf_path: str) -> dict:
    document = fitz.open(pdf_path)

    # ---------------------------------------------------------
    # 1. READ EMBEDDED PDF METADATA
    # ---------------------------------------------------------

    raw_metadata = document.metadata or {}

    metadata = {
        "title": _clean_metadata_value(
            raw_metadata.get("title")
        ),
        "author": _clean_metadata_value(
            raw_metadata.get("author")
        ),
        "subject": _clean_metadata_value(
            raw_metadata.get("subject")
        ),
        "keywords": _clean_metadata_value(
            raw_metadata.get("keywords")
        ),
    }

    pages = []

    # ---------------------------------------------------------
    # 2. EXTRACT PAGE TEXT
    # ---------------------------------------------------------

    for page_number, page in enumerate(
        document,
        start=1,
    ):
        raw_text = page.get_text("text")

        text = clean_pdf_text(
            raw_text
        )

        pages.append(
            {
                "page_number": page_number,
                "text": text,
            }
        )

    document.close()

    # ---------------------------------------------------------
    # 3. BUILD FULL TEXT
    # ---------------------------------------------------------

    full_text = "\n\n".join(
        page["text"]
        for page in pages
        if page["text"]
    )

    # ---------------------------------------------------------
    # 4. FALLBACK AUTHOR EXTRACTION
    #
    # Embedded metadata is preferred.
    # If missing, inspect first-page text.
    # ---------------------------------------------------------

    if not _clean_metadata_value(
        metadata.get("author")
    ):
        first_page_text = (
            pages[0]["text"]
            if pages
            else ""
        )

        detected_authors = (
            _extract_authors_from_first_page(
                first_page_text,
                metadata.get("title"),
            )
        )

        if detected_authors:
            metadata["author"] = detected_authors

    # ---------------------------------------------------------
    # 5. FALLBACK TITLE EXTRACTION
    #
    # If embedded title is missing, use the first meaningful
    # line of the first page.
    # ---------------------------------------------------------

    if not _clean_metadata_value(
        metadata.get("title")
    ):
        if pages:
            first_lines = [
                line.strip()
                for line in pages[0]["text"].splitlines()
                if line.strip()
            ]

            if first_lines:
                possible_title = first_lines[0]

                if (
                    len(possible_title) <= 300
                    and possible_title.lower()
                    not in {
                        "abstract",
                        "introduction",
                    }
                ):
                    metadata["title"] = (
                        possible_title
                    )

    # ---------------------------------------------------------
    # 6. OCR CHECK
    # ---------------------------------------------------------

    ocr_required = needs_ocr(
        full_text,
        len(pages),
    )

    return {
        "full_text": full_text,
        "pages": pages,
        "page_count": len(pages),
        "needs_ocr": ocr_required,
        "metadata": metadata,
    }