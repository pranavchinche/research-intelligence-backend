import asyncio

from app.core.database import AsyncSessionlocal
from app.services.embedding_storage_service import store_chunk_embeddings


async def main():
    chunks = [
        {
            "chunk_id": 0,
            "page_number": 1,
            "text": "Transformers use self-attention mechanisms to process relationships between tokens.",
        },
        {
            "chunk_id": 1,
            "page_number": 1,
            "text": "Attention allows a model to assign different importance to different tokens.",
        },
    ]

    async with AsyncSessionlocal() as db:
        count = await store_chunk_embeddings(
            db=db,
            paper_id=1,
            chunks=chunks,
            model_name="sentence-transformers",
        )

        print("Stored chunks:", count)


if __name__ == "__main__":
    asyncio.run(main())