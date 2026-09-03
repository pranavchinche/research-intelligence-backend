import asyncio

from sqlalchemy import select
from app.core.database import AsyncSessionlocal
from app.models.paper_chunk import PaperChunk


async def main():

    async with AsyncSessionlocal() as db:

        result = await db.execute(
            select(PaperChunk)
            .where(
                PaperChunk.text.ilike("%optimizer%")
                | PaperChunk.text.ilike("%Adam%")
                | PaperChunk.text.ilike("%learning rate%")
            )
        )

        chunks = result.scalars().all()

        print("\nMATCHING CHUNKS")
        print("=" * 80)

        for chunk in chunks:
            print(
                f"\nPaper ID: {chunk.paper_id}"
                f"\nChunk ID: {chunk.chunk_id}"
                f"\nPage: {chunk.page_number}"
            )

            print("-" * 80)
            print(chunk.text)


if __name__ == "__main__":
    asyncio.run(main())