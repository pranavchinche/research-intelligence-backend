import asyncio

from app.core.database import AsyncSessionlocal
from app.services.rag.retriever import retrieve_chunks


async def main():

    async with AsyncSessionlocal() as db:

        question = "What optimizer and learning rate were used to train the Transformer?"

        results = await retrieve_chunks(
            db=db,
            query=question,
            top_k=10,
            min_similarity=0.0,
        )

        print("\nRAW RETRIEVAL")
        print("=" * 80)

        for i, result in enumerate(results, start=1):

            print(f"\nRESULT {i}")
            print("-" * 80)

            print(
                f"Paper ID: {result['paper_id']}\n"
                f"Chunk ID: {result['chunk_id']}\n"
                f"Page: {result['page_number']}\n"
                f"Similarity: {result['similarity']:.4f}"
            )

            print("\nTEXT:")
            print(result["text"])


if __name__ == "__main__":
    asyncio.run(main())