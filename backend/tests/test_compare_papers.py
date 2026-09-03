import asyncio
import hashlib

from sqlalchemy import select

from app.core.database import AsyncSessionlocal
from app.models.paper import Paper
from app.models.paper_chunk import PaperChunk


async def main():

    async with AsyncSessionlocal() as db:

        for paper_id in [3, 4]:

            result = await db.execute(
                select(Paper).where(
                    Paper.id == paper_id
                )
            )

            paper = result.scalar_one_or_none()

            if paper is None:
                print(f"Paper {paper_id} not found")
                continue

            chunks_result = await db.execute(
                select(PaperChunk)
                .where(
                    PaperChunk.paper_id == paper_id
                )
                .order_by(PaperChunk.chunk_id)
            )

            chunks = chunks_result.scalars().all()

            print("=" * 70)
            print(f"PAPER {paper_id}")
            print("=" * 70)

            print("Title:", paper.title)
            print("Source:", paper.source)
            print("Source ID:", paper.source_id)
            print("DOI:", paper.doi)
            print("Full text length:", len(paper.full_text or ""))
            print("Chunks:", len(chunks))

            text = paper.full_text or ""

            text_hash = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()

            print("Full text SHA256:", text_hash)

            if chunks:
                print(
                    "First chunk:",
                    chunks[0].text[:500]
                )

                chunk_hash = hashlib.sha256(
                    chunks[0].text.encode("utf-8")
                ).hexdigest()

                print(
                    "First chunk SHA256:",
                    chunk_hash
                )


asyncio.run(main())