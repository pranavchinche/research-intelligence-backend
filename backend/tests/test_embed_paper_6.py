import asyncio

from app.core.database import AsyncSessionlocal
from app.services.embedding_pipeline import generate_chunk_embeddings


async def main():

    async with AsyncSessionlocal() as db:

        count = await generate_chunk_embeddings(
            db=db,
            paper_id=6,
        )

        print("=" * 70)
        print("PAPER 6 EMBEDDING RESULT")
        print("=" * 70)
        print(f"Embeddings generated: {count}")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())