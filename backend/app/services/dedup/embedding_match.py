"""Embedding-similarity-based duplicate detection.

Catches preprint vs. camera-ready or near-identical resubmissions
via cosine similarity of abstract embeddings. Medium-low confidence
— flag as possible duplicate, never auto-merge.
"""

import hashlib

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.paper import Paper


# Threshold above which two abstracts are considered
# near-duplicates (not auto-merged, but flagged).
EMBEDDING_SIMILARITY_THRESHOLD = 0.97


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def find_embedding_duplicates(
    db: AsyncSession,
    paper_id: int,
) -> list[dict]:
    """Find papers with near-identical abstract embeddings.

    Returns a list of candidate duplicates with similarity scores.
    """
    from app.services.embedding_service import generate_embedding_async

    stmt = select(Paper).where(Paper.id == paper_id)
    result = await db.execute(stmt)
    paper = result.scalar_one_or_none()

    if not paper or not paper.abstract:
        return []

    try:
        embedding = await generate_embedding_async(paper.abstract)
    except Exception:
        return []

    if not embedding:
        return []

    all_papers_stmt = select(Paper).where(Paper.id != paper_id)
    all_result = await db.execute(all_papers_stmt)
    other_papers = list(all_result.scalars().all())

    duplicates = []
    for other in other_papers:
        if not other.abstract:
            continue
        try:
            other_embedding = await generate_embedding_async(
                other.abstract
            )
        except Exception:
            continue
        if not other_embedding:
            continue

        sim = _cosine_similarity(embedding, other_embedding)
        if sim >= EMBEDDING_SIMILARITY_THRESHOLD:
            duplicates.append({
                "paper_id": other.id,
                "title": other.title,
                "similarity": round(sim, 4),
                "signal": "embedding_similarity",
            })

    duplicates.sort(key=lambda x: x["similarity"], reverse=True)
    return duplicates[:10]
