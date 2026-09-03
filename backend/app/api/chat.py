from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.rag.rag_service import ask_rag


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


class HistoryTurn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=6000)


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    paper_id: int | None = None

    top_k: int = Field(
        default=8,
        ge=1,
        le=20,
    )

    history: list[HistoryTurn] = Field(
        default_factory=list,
        max_length=10,
    )


@router.post("")
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    return await ask_rag(
        db=db,
        question=request.question,
        paper_id=request.paper_id,
        top_k=request.top_k,
        history=[turn.model_dump() for turn in request.history],
    )