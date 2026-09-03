from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.papers import PaperCreate, PaperResponse
from app.services.paper_service import PaperService


router = APIRouter(
    prefix="/papers",
    tags=["Papers"],
)


@router.post(
    "/",
    response_model=PaperResponse
)
async def create_paper(
    paper: PaperCreate,
    db: AsyncSession = Depends(get_db)
):

    return await PaperService.create_paper(
        db,
        paper
    )


@router.get(
    "/",
    response_model=list[PaperResponse]
)
async def get_papers(
    db: AsyncSession = Depends(get_db)
):

    return await PaperService.get_all_papers(db)


@router.get(
    "/{paper_id}",
    response_model=PaperResponse
)
async def get_paper(
    paper_id: int,
    db: AsyncSession = Depends(get_db)
):

    paper = await PaperService.get_paper(
        db,
        paper_id
    )

    if paper is None:
        raise HTTPException(
            status_code=404,
            detail="Paper not found"
        )

    return paper