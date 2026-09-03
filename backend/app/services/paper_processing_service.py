from sqlalchemy.ext.asyncio import AsyncSession

from app.services.paper_ingestion import ingest_paper


async def process_paper(
    db: AsyncSession,
    raw_paper: dict,
    source: str,
):
    return await ingest_paper(
        db=db,
        raw_paper=raw_paper,
        source=source,
    )