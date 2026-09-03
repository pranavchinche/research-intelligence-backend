import asyncio

from app.core.database import AsyncSessionlocal
from app.services.rag.rag_service import ask_rag


async def main():
    async with AsyncSessionlocal() as db:

        result = await ask_rag(
            db=db,
            question = "What architecture does the paper use?",
            paper_id=5,
            top_k=8,
        )

        print("=" * 70)
        print("RAG TEST")
        print("=" * 70)

        print("\nQUESTION:")
        print(result["question"])

        print("\nANSWER:")
        print(result["answer"])

        print("\nCITATION VERIFICATION:")
        print(result["citation_verification"])

        print("\nSOURCES:")
        for source in result["sources"]:
            print(
                f"\nSOURCE {result['sources'].index(source) + 1}"
            )
            print(
                f"Chunk: {source.get('chunk_id')}"
            )
            print(
                f"Page: {source.get('page_number')}"
            )
            print(
                f"Similarity: {source.get('similarity')}"
            )
            print(
                source.get("text", "")[:500]
            )


if __name__ == "__main__":
    asyncio.run(main())