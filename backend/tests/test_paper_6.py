import asyncio

from sqlalchemy import select

from app.core.database import AsyncSessionlocal
from app.models.paper import Paper


async def main():

    async with AsyncSessionlocal() as db:

        result = await db.execute(
            select(Paper).where(Paper.id == 6)
        )

        paper = result.scalar_one_or_none()

        if paper is None:
            print("Paper 6 not found.")
            return

        print("=" * 70)
        print("PAPER 6")
        print("=" * 70)

        print("ID:", paper.id)
        print("Title:", paper.title)
        print("Source:", paper.source)
        print("Source ID:", paper.source_id)
        print("PDF URL:", paper.pdf_url)
        print("PDF PATH:", paper.pdf_path)

        if paper.full_text:
            print("Full text length:", len(paper.full_text))
        else:
            print("Full text: EMPTY")

        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())