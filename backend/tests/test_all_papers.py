import asyncio

from sqlalchemy import select
from app.core.database import AsyncSessionlocal
from app.models.paper import Paper


async def main():

    async with AsyncSessionlocal() as db:

        result = await db.execute(
            select(Paper).order_by(Paper.id)
        )

        papers = result.scalars().all()

        print("\nALL PAPERS")
        print("=" * 80)

        for paper in papers:
            print(
                f"ID: {paper.id} | "
                f"Title: {paper.title} | "
                f"PDF: {paper.pdf_path}"
            )


if __name__ == "__main__":
    asyncio.run(main())