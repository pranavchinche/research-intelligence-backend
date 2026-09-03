import asyncio

from sqlalchemy import select
from app.models.paper_chunk import PaperChunk
from app.core.database import AsyncSessionlocal
from app.models.paper import Paper
from app.services.parsing.pdf_parser import extract_pdf_text
from app.services.chunking.chunker import chunk_pages
from app.services.chunking.chunk_repository import create_paper_chunks
from app.services.embedding_pipeline import generate_chunk_embeddings


async def main():

    async with AsyncSessionlocal() as db:

        # --------------------------------------------------
        # 1. Get existing paper
        # --------------------------------------------------

        result = await db.execute(
            select(Paper).limit(1)
        )

        paper = result.scalar_one_or_none()

        if paper is None:
            print("ERROR: No paper found.")
            return

        print("=" * 80)
        print("PAPER")
        print("=" * 80)
        print("Paper ID:", paper.id)
        print("Title:", paper.title)
        print("PDF path:", paper.pdf_path)

        if not paper.pdf_path:
            print("ERROR: Paper has no pdf_path.")
            return

        # --------------------------------------------------
        # 2. Delete existing chunks
        # --------------------------------------------------

        print("\n" + "=" * 80)
        print("DELETING OLD CHUNKS")
        print("=" * 80)

        delete_result = await db.execute(
            select(PaperChunk).where(
                PaperChunk.paper_id == paper.id
            )
        )

        old_chunks = delete_result.scalars().all()

        print("Old chunks:", len(old_chunks))

        for chunk in old_chunks:
            await db.delete(chunk)

        await db.commit()

        print("Old chunks deleted.")

        # --------------------------------------------------
        # 3. Extract PDF
        # --------------------------------------------------

        print("\n" + "=" * 80)
        print("EXTRACTING PDF")
        print("=" * 80)

        parsed = extract_pdf_text(
            paper.pdf_path
        )

        print("Pages:", parsed["page_count"])
        print("Needs OCR:", parsed["needs_ocr"])

        if parsed["needs_ocr"]:
            print("ERROR: PDF requires OCR.")
            return

        # --------------------------------------------------
        # 4. Chunk using the ONLY chunker
        # --------------------------------------------------

        print("\n" + "=" * 80)
        print("CREATING CHUNKS")
        print("=" * 80)

        chunks = chunk_pages(
            parsed["pages"]
        )

        print("Chunks generated:", len(chunks))

        if not chunks:
            print("ERROR: No chunks generated.")
            return

        # --------------------------------------------------
        # 5. Verify chunk IDs
        # --------------------------------------------------

        print("\nFirst 10 chunks:")

        for chunk in chunks[:10]:

            print(
                f"Chunk {chunk['chunk_id']} | "
                f"Page {chunk['page_number']} | "
                f"Length {len(chunk['text'])}"
            )

        # --------------------------------------------------
        # 6. Save chunks
        # --------------------------------------------------

        print("\n" + "=" * 80)
        print("SAVING CHUNKS")
        print("=" * 80)

        count = await create_paper_chunks(
            db,
            paper.id,
            chunks
        )

        print("Chunks inserted:", count)

        # --------------------------------------------------
        # 7. Generate embeddings
        # --------------------------------------------------

        print("\n" + "=" * 80)
        print("GENERATING EMBEDDINGS")
        print("=" * 80)

        embedding_count = await generate_chunk_embeddings(
            db,
            paper.id
        )

        print(
            "Embeddings generated:",
            embedding_count
        )

        # --------------------------------------------------
        # 8. Final verification
        # --------------------------------------------------

        result = await db.execute(
            select(PaperChunk)
            .where(
                PaperChunk.paper_id == paper.id
            )
            .order_by(
                PaperChunk.chunk_id
            )
        )

        final_chunks = result.scalars().all()

        print("\n" + "=" * 80)
        print("FINAL DATABASE CHECK")
        print("=" * 80)

        print(
            "Total chunks:",
            len(final_chunks)
        )

        embedding_exists = sum(
            1
            for chunk in final_chunks
            if chunk.embedding is not None
        )

        print(
            "Chunks with embeddings:",
            embedding_exists
        )

        # --------------------------------------------------
        # 9. Check duplicate chunk IDs
        # --------------------------------------------------

        chunk_ids = [
            chunk.chunk_id
            for chunk in final_chunks
        ]

        unique_ids = set(chunk_ids)

        print(
            "Unique chunk IDs:",
            len(unique_ids)
        )

        if len(chunk_ids) != len(unique_ids):

            print(
                "WARNING: Duplicate chunk IDs detected!"
            )

        else:

            print(
                "SUCCESS: Chunk IDs are unique."
            )

        print("\nPipeline test completed.")


if __name__ == "__main__":
    asyncio.run(main())