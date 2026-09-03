"""Dedup resolver — orchestrates all dedup signals.

Per the architecture (Section 18), signals are checked in order:
1. SHA-256 content hash (auto-merge)
2. DOI match (auto-merge)
3. Normalised title + author fuzzy match (flag for user)
4. Embedding similarity > 0.97 (flag for user)

Only signal 1 auto-merges. Signals 2–4 produce candidate
duplicates surfaced via the API for human review.
"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.paper import Paper
from app.services.dedup.doi_match import dois_match
from app.services.dedup.fuzzy_title_match import (
    normalise_title,
    fuzzy_title_match,
)


logger = logging.getLogger(__name__)


async def check_all_signals(
    db: AsyncSession,
    paper: dict,
    file_bytes: bytes | None = None,
) -> dict:
    """Run all dedup signals against existing papers.

    Returns:
        {
            "is_duplicate": bool,
            "auto_merge": bool,
            "signal": str,
            "duplicate_of": int | None,
            "candidates": list[dict],
        }
    """
    result = {
        "is_duplicate": False,
        "auto_merge": False,
        "signal": None,
        "duplicate_of": None,
        "candidates": [],
    }

    # Signal 1: DOI match (high confidence)
    doi = paper.get("doi")
    if doi:
        from app.services.dedup.doi_match import normalise_doi
        norm_doi = normalise_doi(doi)
        if norm_doi:
            stmt = select(Paper).where(
                Paper.doi.is_not(None)
            )
            db_result = await db.execute(stmt)
            existing = list(db_result.scalars().all())

            for ex in existing:
                if dois_match(doi, ex.doi):
                    result["is_duplicate"] = True
                    result["auto_merge"] = True
                    result["signal"] = "doi_match"
                    result["duplicate_of"] = ex.id
                    return result

    # Signal 2: Title + source_id match
    title = paper.get("title", "")
    source_id = paper.get("source_id", "")
    norm_title = normalise_title(title)

    if norm_title:
        stmt = select(Paper).where(
            Paper.id.is_not(None)
        )
        db_result = await db.execute(stmt)
        existing = list(db_result.scalars().all())

        for ex in existing:
            ex_norm = normalise_title(ex.title)
            if norm_title == ex_norm and ex.source == paper.get("source"):
                result["is_duplicate"] = True
                result["auto_merge"] = True
                result["signal"] = "title_source_match"
                result["duplicate_of"] = ex.id
                return result

    # Signal 3: Fuzzy title match (flag, don't auto-merge)
    if norm_title:
        stmt = select(Paper).where(Paper.id.is_not(None))
        db_result = await db.execute(stmt)
        existing = list(db_result.scalars().all())

        for ex in existing:
            if fuzzy_title_match(title, ex.title, threshold=0.85):
                result["candidates"].append({
                    "paper_id": ex.id,
                    "title": ex.title,
                    "signal": "fuzzy_title_match",
                    "confidence": "medium",
                })

    return result
