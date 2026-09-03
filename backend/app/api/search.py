#D:\FYP\main\backend\app\api\search.py

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.paper_sources.arxiv import search_arxiv
from app.services.paper_sources.openalex import search_openalex
from app.core.database import get_db
from app.services.paper_ingestion import ingest_paper

router = APIRouter(
    prefix="/search",
    tags=["Search"],
)


class IngestRequest(BaseModel):
    paper: dict
    source: str = Field(min_length=1)


@router.get("/arxiv")
async def search_arxiv_papers(
    query: str,
    max_results: int = Query(default=5, ge=1, le=50),
):
    return await search_arxiv(query, max_results)


@router.get("/arvix")
async def search_arxiv_legacy(
    query: str,
    max_results: int = Query(default=5, ge=1, le=50),
):
    return await search_arxiv(query, max_results)

@router.get("/openalex")
async def search_openalex_papers(
    query: str,
    max_results: int = Query(default=5, ge=1, le=50),
):
    return await search_openalex(
        query,
        max_results
    )


@router.post("/ingest")
async def ingest_search_result(
    request: IngestRequest,
    db: AsyncSession = Depends(get_db)
):
    paper = request.paper

    # Support both:
    #
    # {
    #   "title": "...",
    #   ...
    # }
    #
    # and:
    #
    # {
    #   "paper": {
    #       "title": "...",
    #       ...
    #   }
    # }

    if "paper" in paper and isinstance(paper["paper"], dict):
        paper = paper["paper"]

    return await ingest_paper(
        db,
        paper,
        request.source
    )
