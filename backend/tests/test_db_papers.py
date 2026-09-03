import asyncio

from sqlalchemy import select, func

from app.core.database import AsyncSessionlocal
from app.models.paper_chunk import PaperChunk


async def main():

    async with AsyncSessionlocal() as db:

        result = await db.execute(
            select(
                PaperChunk.paper_id,
                func.count(PaperChunk.id)
            )
            .where(PaperChunk.embedding.is_not(None))
            .group_by(PaperChunk.paper_id)
            .order_by(PaperChunk.paper_id)
        )

        rows = result.all()

        print("\nPAPERS WITH EMBEDDINGS")
        print("=" * 50)

        for paper_id, count in rows:
            print(
                f"Paper ID: {paper_id} | "
                f"Chunks: {count}"
            )


if __name__ == "__main__":
    asyncio.run(main())