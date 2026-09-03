import asyncio

from app.core.database import AsyncSessionlocal
from app.services.paper_ingestion import ingest_paper


async def main():

    paper = {
        "title": "Seq2Seq Research Paper",
        "authors": [],
        "abstract": "",
        "doi": None,
        "source": "local",
        "source_id": "seq2seq-local",
        "arxiv_id": None,
        "published_date": None,
        "updated_date": None,
        "categories": [],
        "pdf_url": None,

        # IMPORTANT:
        # This is the actual PDF we already have.
        "pdf_path": r"D:\FYP\main\backend\uploads\seq2seq.pdf",
    }

    async with AsyncSessionlocal() as db:

        result = await ingest_paper(
            db=db,
            raw_paper=paper,
            source="local",
        )

        print("\nINGESTION RESULT")
        print("=" * 60)
        print(result)


if __name__ == "__main__":
    asyncio.run(main())