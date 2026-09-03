from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.summary.summary_service import (
    generate_executive_summary,
)


router = APIRouter(
    prefix="/summary",
    tags=["Executive Summary"],
)


@router.get("/{paper_id}")
async def get_executive_summary(
    paper_id: int,
    db: AsyncSession = Depends(get_db),
):

    result = await generate_executive_summary(
        db=db,
        paper_id=paper_id,
    )

    if result.get("error") == "Paper not found":
        raise HTTPException(
            status_code=404,
            detail="Paper not found",
        )

    return result
