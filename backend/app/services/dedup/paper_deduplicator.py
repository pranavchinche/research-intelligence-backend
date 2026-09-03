import hashlib
import re

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.paper import Paper


def normalize_title(title: str | None) -> str:
    """
    Normalize a paper title for duplicate detection.

    Removes:
    - file extensions
    - punctuation
    - extra whitespace
    - common PDF/upload naming noise
    """

    if not title:
        return ""

    title = title.lower().strip()

    # Remove common file extensions.
    title = re.sub(
        r"\.(pdf|txt)$",
        "",
        title,
        flags=re.IGNORECASE,
    )

    # Remove common upload/file naming noise.
    title = re.sub(
        r"\b(research\s+paper|paper)\b",
        "",
        title,
    )

    # Keep letters, numbers and spaces.
    title = re.sub(
        r"[^a-z0-9\s]",
        " ",
        title,
    )

    # Normalize whitespace.
    title = re.sub(
        r"\s+",
        " ",
        title,
    )

    return title.strip()


def normalize_doi(doi: str | None) -> str | None:
    """
    Normalize DOI values so harmless formatting differences
    do not create duplicate papers.
    """

    if not doi:
        return None

    doi = doi.strip().lower()

    doi = re.sub(
        r"^https?://doi\.org/",
        "",
        doi,
    )

    doi = re.sub(
        r"^doi:\s*",
        "",
        doi,
    )

    return doi.strip()


def normalize_arxiv_id(arxiv_id: str | None) -> str | None:
    """
    Normalize arXiv identifiers.
    """

    if not arxiv_id:
        return None

    arxiv_id = arxiv_id.strip().lower()

    arxiv_id = re.sub(
        r"^https?://arxiv\.org/abs/",
        "",
        arxiv_id,
    )

    arxiv_id = re.sub(
        r"^arxiv:",
        "",
        arxiv_id,
    )

    return arxiv_id.strip()


def calculate_text_hash(text: str | None) -> str | None:
    """
    Calculate SHA256 hash of full paper text.
    """

    if not text:
        return None

    normalized_text = text.strip()

    if not normalized_text:
        return None

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()


async def find_duplicate(
    db: AsyncSession,
    paper_data: dict,
) -> Paper | None:
    """
    Find an existing paper that represents the same paper.

    Detection order:

    1. DOI
    2. arXiv ID
    3. source + source ID
    4. exact full-text hash
    5. normalized title
    """

    # ---------------------------------------------------------
    # 1. DOI MATCH
    # ---------------------------------------------------------

    doi = normalize_doi(
        paper_data.get("doi")
    )

    if doi:

        result = await db.execute(
            select(Paper).where(
                Paper.doi == doi
            )
        )

        existing = result.scalar_one_or_none()

        if existing:
            return existing

    # ---------------------------------------------------------
    # 2. ARXIV ID MATCH
    # ---------------------------------------------------------

    arxiv_id = normalize_arxiv_id(
        paper_data.get("arxiv_id")
    )

    if arxiv_id:

        result = await db.execute(
            select(Paper).where(
                Paper.arxiv_id == arxiv_id
            )
        )

        existing = result.scalar_one_or_none()

        if existing:
            return existing

    # ---------------------------------------------------------
    # 3. SOURCE + SOURCE ID MATCH
    # ---------------------------------------------------------

    source = paper_data.get("source")
    source_id = paper_data.get("source_id")

    if source and source_id:

        result = await db.execute(
            select(Paper).where(
                Paper.source == source,
                Paper.source_id == source_id,
            )
        )

        existing = result.scalar_one_or_none()

        if existing:
            return existing

    # ---------------------------------------------------------
    # 4. EXACT FULL-TEXT HASH MATCH
    # ---------------------------------------------------------

    full_text = paper_data.get("full_text")

    text_hash = calculate_text_hash(
        full_text
    )

    if text_hash:

        BATCH = 200
        offset = 0

        while True:
            result = await db.execute(
                select(Paper).where(
                    Paper.full_text.is_not(None)
                )
                .offset(offset)
                .limit(BATCH)
            )

            batch = result.scalars().all()

            if not batch:
                break

            for existing in batch:

                existing_hash = calculate_text_hash(
                    existing.full_text
                )

                if existing_hash == text_hash:
                    return existing

            offset += BATCH

    # ---------------------------------------------------------
    # 5. NORMALIZED TITLE MATCH (batched, memory-safe)
    # ---------------------------------------------------------

    title = normalize_title(
        paper_data.get("title")
    )

    if title:

        BATCH = 200
        offset = 0

        while True:
            result = await db.execute(
                select(Paper)
                .offset(offset)
                .limit(BATCH)
            )

            batch = result.scalars().all()

            if not batch:
                break

            for existing in batch:

                existing_title = normalize_title(
                    existing.title
                )

                if (
                    existing_title
                    and existing_title == title
                ):
                    return existing

            offset += BATCH

    return None