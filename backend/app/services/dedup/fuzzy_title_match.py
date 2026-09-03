"""Fuzzy title + author matching for duplicate detection.

Catches same paper with slightly different metadata formatting.
Medium confidence — flag for user confirmation, don't auto-merge.
"""

import re


def normalise_title(title: str | None) -> str:
    """Normalise a title for fuzzy comparison."""
    if not title:
        return ""
    title = title.lower().strip()
    title = re.sub(r"\.(pdf|txt)$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"[^a-z0-9\s]", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def normalise_authors(authors: list[str] | str | None) -> str:
    """Normalise author list to a sorted, lowercased string."""
    if not authors:
        return ""
    if isinstance(authors, str):
        parts = [a.strip().lower() for a in authors.split(",")]
    else:
        parts = [a.strip().lower() for a in authors]
    surnames = []
    for a in parts:
        tokens = a.split()
        if tokens:
            surnames.append(tokens[-1])
    surnames.sort()
    return "|".join(surnames)


def fuzzy_title_match(
    title_a: str | None,
    title_b: str | None,
    threshold: float = 0.85,
) -> bool:
    """Check if two titles are fuzzy-identical.

    Uses a simple character overlap ratio (Jaccard on character
    trigrams) rather than requiring a library.
    """
    a = normalise_title(title_a)
    b = normalise_title(title_b)

    if not a or not b:
        return False

    if a == b:
        return True

    # Trigram Jaccard similarity
    def trigrams(s: str) -> set:
        return {s[i : i + 3] for i in range(len(s) - 2)}

    tri_a = trigrams(a)
    tri_b = trigrams(b)

    if not tri_a or not tri_b:
        return False

    intersection = tri_a & tri_b
    union = tri_a | tri_b
    similarity = len(intersection) / len(union)

    return similarity >= threshold


def title_author_match(
    title_a: str | None,
    authors_a: list[str] | str | None,
    title_b: str | None,
    authors_b: list[str] | str | None,
) -> bool:
    """Combined title + author matching.

    Returns True when titles are fuzzy-similar AND authors overlap.
    """
    if not fuzzy_title_match(title_a, title_b):
        return False

    auth_a = normalise_authors(authors_a)
    auth_b = normalise_authors(authors_b)

    if not auth_a or not auth_b:
        return False

    return auth_a == auth_b
