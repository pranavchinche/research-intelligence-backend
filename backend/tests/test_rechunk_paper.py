import asyncio

from sqlalchemy import delete, select
from app.core.database import AsyncSessionlocal
from app.models.paper_chunk import PaperChunk


PAPER_ID = 1


async def main():

    async with AsyncSessionlocal() as db:

        # --------------------------------------------------
        # 1. Check existing chunks
        # --------------------------------------------------

        result = await db.execute(
            select(PaperChunk)
            .where(PaperChunk.paper_id == PAPER_ID)
            .order_by(PaperChunk.chunk_id)
        )

        chunks = result.scalars().all()

        print("\nBEFORE DELETE")
        print("=" * 80)
        print(f"Paper ID: {PAPER_ID}")
        print(f"Existing chunks: {len(chunks)}")

        for chunk in chunks[:10]:
            print(
                f"Chunk {chunk.chunk_id} | "
                f"Page {chunk.page_number}"
            )

        if len(chunks) > 10:
            print(f"... and {len(chunks) - 10} more")

        # --------------------------------------------------
        # 2. Delete all chunks for this paper
        # --------------------------------------------------

        stmt = delete(PaperChunk).where(
            PaperChunk.paper_id == PAPER_ID
        )

        result = await db.execute(stmt)

        deleted_count = result.rowcount or 0

        await db.commit()

        print("\nDELETE RESULT")
        print("=" * 80)
        print(f"Deleted chunks: {deleted_count}")

        # --------------------------------------------------
        # 3. Verify deletion
        # --------------------------------------------------

        result = await db.execute(
            select(PaperChunk)
            .where(PaperChunk.paper_id == PAPER_ID)
        )

        remaining_chunks = result.scalars().all()

        print("\nAFTER DELETE")
        print("=" * 80)
        print(f"Remaining chunks: {len(remaining_chunks)}")

        if len(remaining_chunks) == 0:
            print("SUCCESS: All chunks deleted.")
        else:
            print("WARNING: Some chunks still remain.")


if __name__ == "__main__":
    asyncio.run(main())