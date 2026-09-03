from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.novelty.scorer import calculate_novelty


router = APIRouter(
    prefix="/novelty",
    tags=["Novelty"],
)


@router.get("/{paper_id}")
async def get_novelty(
    paper_id: int,
    db: AsyncSession = Depends(get_db),
):
    return await calculate_novelty(
        db=db,
        paper_id=paper_id,
    )