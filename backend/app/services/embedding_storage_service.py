from sqlalchemy.ext.asyncio import AsyncSession

from app.models.paper_chunk import PaperChunk
from app.services.embedding_service import generate_embedding_async


async def store_chunk_embeddings(
    db: AsyncSession,
    paper_id: int,
    chunks: list[dict],
    model_name: str,
) -> int:
    stored_count = 0

    for chunk in chunks:
        text = chunk["text"]

        embedding = await generate_embedding_async(text)

        paper_chunk = PaperChunk(
            paper_id=paper_id,
            chunk_id=chunk["chunk_id"],
            page_number=chunk.get("page_number"),
            text=text,
            model=model_name,
            embedding=embedding,
        )

        db.add(paper_chunk)
        stored_count += 1

    await db.commit()

    return stored_count