"""DOI-based duplicate detection.

Same published work, different file (e.g. re-downloaded PDF).
High-confidence signal — auto-merge when DOI matches exactly.
"""

import re


def normalise_doi(doi: str | None) -> str | None:
    """Normalise a DOI for comparison."""
    if not doi:
        return None
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.strip() or None


def dois_match(doi_a: str | None, doi_b: str | None) -> bool:
    """Check if two DOIs refer to the same work."""
    a = normalise_doi(doi_a)
    b = normalise_doi(doi_b)
    if not a or not b:
        return False
    return a == b
