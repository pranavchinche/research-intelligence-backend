import asyncio

from sqlalchemy import select, delete
from app.core.database import AsyncSessionlocal
from app.models.paper import Paper
from app.models.paper_chunk import PaperChunk


CANONICAL_PAPER_ID = 3
DUPLICATE_PAPER_ID = 4


async def cleanup():
    async with AsyncSessionlocal() as db:

        # --------------------------------------------------
        # Verify both papers exist
        # --------------------------------------------------

        result = await db.execute(
            select(Paper).where(
                Paper.id.in_([
                    CANONICAL_PAPER_ID,
                    DUPLICATE_PAPER_ID
                ])
            )
        )

        papers = {
            paper.id: paper
            for paper in result.scalars().all()
        }

        if CANONICAL_PAPER_ID not in papers:
            print("ERROR: Canonical paper 3 was not found.")
            return

        if DUPLICATE_PAPER_ID not in papers:
            print("Paper 4 does not exist. Nothing to clean.")
            return

        canonical = papers[CANONICAL_PAPER_ID]
        duplicate = papers[DUPLICATE_PAPER_ID]

        print("=" * 70)
        print("DUPLICATE CLEANUP")
        print("=" * 70)

        print(
            f"Keeping Paper {canonical.id}: "
            f"{canonical.title}"
        )

        print(
            f"Deleting Paper {duplicate.id}: "
            f"{duplicate.title}"
        )

        # --------------------------------------------------
        # Delete duplicate chunks first
        # --------------------------------------------------

        result = await db.execute(
            select(PaperChunk).where(
                PaperChunk.paper_id == DUPLICATE_PAPER_ID
            )
        )

        chunks = result.scalars().all()

        print(
            f"Duplicate chunks found: {len(chunks)}"
        )

        for chunk in chunks:
            await db.delete(chunk)

        # --------------------------------------------------
        # Delete duplicate paper
        # --------------------------------------------------

        await db.delete(duplicate)

        await db.commit()

        print()
        print("Cleanup completed successfully.")
        print(
            f"Paper {CANONICAL_PAPER_ID} was preserved."
        )
        print(
            f"Paper {DUPLICATE_PAPER_ID} was deleted."
        )
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(cleanup())