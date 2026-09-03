#D:\FYP\main\backend\app\services\paper_repository.py
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.paper import Paper


class PaperRepository:

    @staticmethod
    async def create(
        db: AsyncSession,
        paper_data: dict
    ) -> Paper:

        paper = Paper(**paper_data)

        db.add(paper)
        await db.commit()
        await db.refresh(paper)

        return paper

    @staticmethod
    async def get_all(
        db: AsyncSession
    ) -> list[Paper]:

        result = await db.execute(
            select(Paper).order_by(Paper.id.desc())
        )

        return list(result.scalars().all())

    @staticmethod
    async def get_by_id(
        db: AsyncSession,
        paper_id: int
    ) -> Paper | None:

        result = await db.execute(
            select(Paper).where(Paper.id == paper_id)
        )

        return result.scalar_one_or_none()