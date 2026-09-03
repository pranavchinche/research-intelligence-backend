from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.models.paper import Paper
from app.models.paper_chunk import PaperChunk
from app.schemas.validation import ValidationResponse
from app.services.paper_validation_service import validate_paper_full


router = APIRouter(
    prefix="/validation",
    tags=["Validation"],
)


@router.get(
    "/{paper_id}",
    response_model=ValidationResponse,
)
async def validate_paper(
    paper_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Paper).where(Paper.id == paper_id)
    )
    paper = result.scalar_one_or_none()

    if paper is None:
        raise HTTPException(
            status_code=404,
            detail="Paper not found",
        )

    chunks_result = await db.execute(
        select(PaperChunk)
        .where(PaperChunk.paper_id == paper_id)
        .order_by(PaperChunk.chunk_id)
    )
    chunks = list(chunks_result.scalars().all())

    validation = validate_paper_full(paper, chunks)

    return validation
