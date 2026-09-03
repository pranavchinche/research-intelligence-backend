import re


# The ONLY citation format we accept as a valid RAG citation:
# [SOURCE 1], [SOURCE 2], etc. (case-sensitive "SOURCE").
#
# Whitespace immediately inside the brackets - "[ SOURCE 6 ]" - is
# tolerated here because it is still unambiguously our citation
# format, just with extra spacing the model sometimes adds. This is
# NOT a loosening of what counts as a citation: the literal word
# "SOURCE" (exact case), the brackets, and the digits are all still
# required exactly as before - only incidental whitespace around
# them is permitted.
CITATION_PATTERN = re.compile(
    r"\[\s*SOURCE\s+(\d+)\s*\]"
)

# A looser, case-insensitive pattern used ONLY to catch near-miss
# attempts at our citation format - things like "(Source 1)",
# "[source 1]", or a bare "SOURCE 1" with no brackets at all. These
# must be flagged as invalid rather than silently ignored, since
# they represent the model attempting (and failing) to use our
# citation format.
#
# This intentionally requires the literal word "source" to appear,
# which means it will NEVER match the paper's own internal
# citations such as "[29]", "[5]", "[2]" - those contain no such
# word and are correctly ignored entirely, per spec.
_LOOSE_CITATION_PATTERN = re.compile(
    r"[\(\[]?\s*source\s+(\d+)\s*[\)\]]?",
    re.IGNORECASE,
)

# Lines that are pure formatting/headers (all caps, digits,
# whitespace, dashes, or "=" separators) carry no factual claim of
# their own, so they should never be required to carry a citation.
_HEADING_LINE_PATTERN = re.compile(
    r"^[A-Z0-9 \-=]+$"
)

INSUFFICIENT_CONTEXT_PHRASE = (
    "The available research context is insufficient to answer this question."
)


def extract_citations(answer: str) -> list[int]:
    """
    Extract SOURCE citation numbers from the LLM answer.

    Only citations using the EXACT required format, [SOURCE X], are
    extracted here. Anything else (wrong case, wrong brackets,
    missing brackets, or the paper's own internal citation numbers
    like [29]) is deliberately NOT matched by this pattern.
    """
    matches = CITATION_PATTERN.findall(answer)

    return sorted(
        set(int(number) for number in matches)
    )


def _find_malformed_citation_numbers(answer: str) -> list[int]:
    """
    Find citation-like references to "source N" that do NOT use the
    exact required [SOURCE X] format - wrong case ("[Source 1]"),
    wrong brackets ("(Source 1)"), or no brackets at all
    ("SOURCE 1").

    These are attempts at our citation format that got the format
    wrong, and must be treated as invalid so the citation-format
    problem is visible instead of silently disappearing.

    This never flags the paper's own internal citations (e.g.
    "[29]") because those do not contain the word "source" at all.
    """

    malformed_numbers = []

    for match in _LOOSE_CITATION_PATTERN.finditer(answer):

        number = match.group(1)

        exact_text = match.group(0)

        # If this exact occurrence already satisfies the strict
        # format (SOURCE, exact case, with or without incidental
        # whitespace right inside the brackets), it is a valid
        # citation, not a malformed one.
        if re.fullmatch(
            rf"\[\s*SOURCE\s+{re.escape(number)}\s*\]",
            exact_text,
        ):
            continue

        malformed_numbers.append(int(number))

    return malformed_numbers


def _strip_heading_lines(answer: str) -> str:
    """
    Remove pure heading/separator lines (e.g. "ANSWER", "======")
    so that a heading with no factual content is never mistaken for
    an uncited factual claim.
    """

    kept_lines = [
        line
        for line in answer.splitlines()
        if not (
            line.strip()
            and _HEADING_LINE_PATTERN.match(line.strip())
        )
    ]

    return "\n".join(kept_lines)


def verify_citations(
    answer: str,
    source_count: int,
) -> dict:
    """
    Verify citations used by the LLM against the required RAG
    citation contract:

    - Citations must use the exact [SOURCE X] format.
    - X must fall within the range of sources actually supplied.
    - A near-miss citation attempt (wrong case/brackets/spacing) is
      invalid, not merely ignored.
    - The paper's own internal citations (e.g. "[29]") are ignored
      entirely - they are not RAG citations at all.
    - An explicit, EXACT insufficient-context response never
      requires a citation, since it makes no factual claim about
      the paper. This must be an exact match, not a substring
      check - otherwise an answer that mixes the insufficient-
      context phrase with additional unsupported claims would be
      wrongly marked valid.
    """

    # ---------------------------------------------------------
    # CASE 1: Model's ENTIRE answer is the exact insufficient-
    # context response (whitespace at the edges is tolerated).
    # ---------------------------------------------------------
    if answer.strip() == INSUFFICIENT_CONTEXT_PHRASE:

        return {
            "valid": True,
            "citations": [],
            "invalid_citations": [],
            "missing_citations": False,
        }

    # ---------------------------------------------------------
    # CASE 2: Normal answer
    # ---------------------------------------------------------

    citations = extract_citations(answer)

    out_of_range_citations = [
        citation
        for citation in citations
        if citation < 1 or citation > source_count
    ]

    malformed_citation_numbers = _find_malformed_citation_numbers(
        answer
    )

    invalid_citations = sorted(
        set(out_of_range_citations)
        | set(malformed_citation_numbers)
    )

    # A citation is "missing" only if there is factual content in
    # the answer (ignoring pure heading/separator lines) with no
    # correctly-formatted [SOURCE X] citation anywhere in it.
    content_only = _strip_heading_lines(answer).strip()

    missing_citations = (
        len(citations) == 0
        and len(content_only) > 0
    )

    return {
        "valid": (
            len(invalid_citations) == 0
            and not missing_citations
        ),
        "citations": citations,
        "invalid_citations": invalid_citations,
        "missing_citations": missing_citations,
    }