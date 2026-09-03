from datetime import timezone

from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.papers import PaperCreate
from app.services.paper_repository import PaperRepository


class PaperService:

    @staticmethod
    async def create_paper(
        db: AsyncSession,
        paper_data: PaperCreate
    ):
        data = paper_data.model_dump()

        # Convert timezone-aware datetime to naive UTC
        # because the database columns use TIMESTAMP WITHOUT TIME ZONE
        for field in ("published_date", "updated_date"):
            if data.get(field) is not None and data[field].tzinfo is not None:
                data[field] = (
                    data[field]
                    .astimezone(timezone.utc)
                    .replace(tzinfo=None)
                )

        return await PaperRepository.create(
            db,
            data
        )

    @staticmethod
    async def get_all_papers(
        db: AsyncSession
    ):
        return await PaperRepository.get_all(db)

    @staticmethod
    async def get_paper(
        db: AsyncSession,
        paper_id: int
    ):
        return await PaperRepository.get_by_id(
            db,
            paper_id
        )