import asyncio

from sqlalchemy import select, func

from app.core.database import AsyncSessionlocal
from app.models.paper_chunk import PaperChunk


async def main():

    async with AsyncSessionlocal() as db:

        result = await db.execute(
            select(func.count(PaperChunk.id))
            .where(PaperChunk.paper_id == 6)
        )

        total_chunks = result.scalar_one()

        result = await db.execute(
            select(func.count(PaperChunk.id))
            .where(
                PaperChunk.paper_id == 6,
                PaperChunk.embedding.is_not(None),
            )
        )

        embedded_chunks = result.scalar_one()

        print("=" * 70)
        print("PAPER 6 CHUNK DIAGNOSTIC")
        print("=" * 70)
        print(f"Total chunks: {total_chunks}")
        print(f"Chunks with embeddings: {embedded_chunks}")
        print(f"Chunks without embeddings: {total_chunks - embedded_chunks}")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())