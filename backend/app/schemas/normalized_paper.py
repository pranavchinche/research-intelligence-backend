#D:\FYP\main\backend\app\schemas\normalized_paper.py

from pydantic import BaseModel, Field


class NormalizedPaper(BaseModel):
    title: str = Field(min_length=1)

    abstract: str | None = None
    authors: list[str] = []

    doi: str | None = None

    source: str
    source_id: str | None = None
    arxiv_id: str | None = None

    published_date: str | None = None
    updated_date: str | None = None

    categories: list[str] = []

    pdf_url: str | None = None