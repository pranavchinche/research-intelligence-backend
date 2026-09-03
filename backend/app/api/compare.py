from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.comparison.comparison_service import (
    compare_papers,
)


router = APIRouter(
    prefix="/compare",
    tags=["Compare"],
)


@router.get("/{paper_id_1}/{paper_id_2}")
async def compare_paper_endpoint(
    paper_id_1: int,
    paper_id_2: int,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await compare_papers(
            db=db,
            paper_id_1=paper_id_1,
            paper_id_2=paper_id_2,
        )

    except ValueError as exc:
        message = str(exc)

        if "not found" in message.lower():
            raise HTTPException(
                status_code=404,
                detail=message,
            )

        if "Cannot compare" in message:
            raise HTTPException(
                status_code=400,
                detail=message,
            )

        if "no embedded chunks" in message.lower():
            raise HTTPException(
                status_code=400,
                detail=message,
            )

        raise HTTPException(
            status_code=400,
            detail=message,
        )