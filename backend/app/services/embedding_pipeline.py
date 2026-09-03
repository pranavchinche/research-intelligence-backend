from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.paper_chunk import PaperChunk
from app.services.embedding_service import generate_embedding_async


async def generate_chunk_embeddings(
    db: AsyncSession,
    paper_id: int,
) -> int:

    result = await db.execute(
        select(PaperChunk)
        .where(
            PaperChunk.paper_id == paper_id,
            PaperChunk.embedding.is_(None),
        )
        .order_by(PaperChunk.id)
    )

    chunks = result.scalars().all()

    if not chunks:
        return 0

    updated_count = 0

    for chunk in chunks:

        text = chunk.text or ""

        # PostgreSQL TEXT / embedding pipeline should never receive
        # NUL characters.
        text = text.replace("\x00", "")

        if not text.strip():
            print(
                f"Skipping chunk {chunk.chunk_id}: empty text"
            )
            continue

        try:
            embedding = await generate_embedding_async(text)

            if not embedding:
                print(
                    f"Skipping chunk {chunk.chunk_id}: "
                    "empty embedding"
                )
                continue

            chunk.embedding = embedding

            updated_count += 1

            print(
                f"Embedded chunk {chunk.chunk_id} "
                f"({len(embedding)} dimensions)"
            )

        except Exception as exc:
            print(
                f"Embedding failed for chunk "
                f"{chunk.chunk_id}: {exc}"
            )

    await db.commit()

    return updated_count