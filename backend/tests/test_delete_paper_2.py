# test_delete_paper_2.py

import asyncio

from sqlalchemy import delete

from app.core.database import AsyncSessionlocal
from app.models.paper import Paper
from app.models.paper_chunk import PaperChunk


async def main():

    async with AsyncSessionlocal() as db:

        # Delete chunks first
        await db.execute(
            delete(PaperChunk).where(
                PaperChunk.paper_id == 2
            )
        )

        # Delete paper
        await db.execute(
            delete(Paper).where(
                Paper.id == 2
            )
        )

        await db.commit()

        print("Paper 2 deleted successfully.")


if __name__ == "__main__":
    asyncio.run(main())