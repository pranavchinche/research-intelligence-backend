import asyncio

from app.services.gaps.limitation_verifier import (
    verify_limitation,
)


async def main():

    text = """
    We used deep LSTMs with 4 layers, with 1000 cells at each layer
    and 1000 dimensional word embeddings.
    """

    result = await verify_limitation(text)

    print()
    print("=" * 60)
    print("LIMITATION VERIFICATION")
    print("=" * 60)

    print("VALID:", result["valid"])
    print("EVIDENCE:", result["evidence"])
    print("RAW:")
    print(result["raw_answer"])


if __name__ == "__main__":
    asyncio.run(main())