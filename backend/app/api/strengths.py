import traceback
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.strengths.strength_pipeline import detect_strengths


router = APIRouter(
    prefix="/strengths",
    tags=["Research Strengths"],
)


@router.get("/{paper_id}")
async def analyze_research_strengths(
    paper_id: int,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await detect_strengths(
            db=db,
            paper_id=paper_id,
            top_k=5,
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n=== STRENGTHS ERROR ===\n{tb}\n=== END ===")
        return {"error": str(e), "traceback": tb}
