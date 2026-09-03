import asyncio

from app.core.database import AsyncSessionlocal
from app.services.gaps.gap_pipeline import detect_research_gaps


async def main():

    async with AsyncSessionlocal() as db:

        result = await detect_research_gaps(
            db=db,
            paper_id=7,
            top_k=5,
        )

        print("\nRESEARCH GAP ANALYSIS")
        print("=" * 80)

        print(result["gaps"])

        print("\nSOURCES")
        print("=" * 80)

        for source in result["sources"]:

            similarity = source.get("similarity")

            if similarity is None:
                similarity_text = "N/A"
            else:
                similarity_text = f"{float(similarity):.4f}"

            print(
                f"Paper ID: {source['paper_id']} | "
                f"Chunk: {source['chunk_id']} | "
                f"Page: {source['page_number']} | "
                f"Similarity: {similarity_text}"
            )


if __name__ == "__main__":
    asyncio.run(main())