import asyncio

from app.core.database import AsyncSessionlocal
from app.services.novelty.scorer import calculate_novelty


async def main():

    async with AsyncSessionlocal() as db:

        result = await calculate_novelty(
            db=db,
            paper_id=3,
        )

        print("\nNOVELTY ANALYSIS")
        print("=" * 80)

        print(
            f"Paper ID: {result['paper_id']}"
        )

        print(
            f"Novelty Score: {result['novelty_score']}"
        )

        print(
            f"Highest Similarity: "
            f"{result.get('highest_similarity')}"
        )

        print("\nCOMPARISONS")
        print("-" * 80)

        for comparison in result["comparisons"]:

            print(
                f"Paper ID: {comparison['paper_id']} | "
                f"Similarity: {comparison['similarity']}"
            )

        if result.get("message"):
            print("\nMESSAGE:")
            print(result["message"])


if __name__ == "__main__":
    asyncio.run(main())