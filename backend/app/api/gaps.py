from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.gaps.gap_pipeline import detect_research_gaps


router = APIRouter(
    prefix="/gaps",
    tags=["Research Gaps"],
)


@router.post("/{paper_id}")
async def analyze_research_gaps(
    paper_id: int,
    db: AsyncSession = Depends(get_db),
):
    return await detect_research_gaps(
        db=db,
        paper_id=paper_id,
        top_k=5,
    )