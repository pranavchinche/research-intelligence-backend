from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.paper_chunk import PaperChunk


async def delete_paper_chunks(
    db: AsyncSession,
    paper_id: int,
) -> int:
    """
    Delete all existing chunks for a paper.

    Used before re-chunking/re-embedding an existing paper.
    """

    stmt = delete(PaperChunk).where(
        PaperChunk.paper_id == paper_id
    )

    result = await db.execute(stmt)

    await db.commit()

    return result.rowcount or 0


async def create_paper_chunks(
    db: AsyncSession,
    paper_id: int,
    chunks: list[dict],
    model_name: str = "all-MiniLM-L6-v2",
) -> int:
    """
    Create chunks for a paper.
    """

    paper_chunks = []

    for chunk in chunks:

        paper_chunk = PaperChunk(
            paper_id=paper_id,
            chunk_id=chunk["chunk_id"],
            page_number=chunk["page_number"],
            text=chunk["text"],
            model=model_name,
        )

        paper_chunks.append(paper_chunk)

    if paper_chunks:

        db.add_all(paper_chunks)

        await db.commit()

    return len(paper_chunks)