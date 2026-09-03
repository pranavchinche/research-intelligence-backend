#D:\FYP\main\backend\app\schemas\papers.py
# D:\FYP\main\backend\app\schemas\papers.py

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PaperCreate(BaseModel):
    """
    Schema used when manually creating a paper through the API.
    """

    title: str

    abstract: str | None = None

    authors: str | None = None

    doi: str | None = None

    source: str

    source_id: str | None = None

    arxiv_id: str | None = None

    published_date: datetime | None = None

    updated_date: datetime | None = None

    categories: str | None = None

    pdf_url: str | None = None

    pdf_path: str | None = None

    drive_file_id: str | None = None

    full_text: str | None = None


class PaperResponse(BaseModel):
    """
    Schema returned by the paper API.
    """

    id: int

    title: str

    abstract: str | None = None

    authors: str | None = None

    doi: str | None = None

    source: str

    source_id: str | None = None

    arxiv_id: str | None = None

    published_date: datetime | None = None

    updated_date: datetime | None = None

    categories: str | None = None

    pdf_url: str | None = None

    pdf_path: str | None = None

    drive_file_id: str | None = None

    full_text: str | None = None

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )