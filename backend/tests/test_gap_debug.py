"""Diagnostic script to trace the gap pipeline for paper_id=1."""
import asyncio
import json
import traceback
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.core.database import AsyncSessionlocal
from app.services.gaps.gap_pipeline import (
    detect_research_gaps,
    validate_gap_answer,
    build_gap_prompt,
    retrieve_chunks,
)
from app.services.rag.retriever import build_context, expand_with_neighbors
from app.services.llm.ollama import generate_answer


async def main():
    paper_id = 1

    async with AsyncSessionlocal() as db:
        print("=" * 80)
        print(f"TESTING GAP PIPELINE FOR PAPER_ID={paper_id}")
        print("=" * 80)

        # Step 1: Test retrieval for a few queries
        print("\n\n=== STEP 1: RETRIEVAL TEST ===")
        gap_queries = [
            "limitations of the proposed method",
            "weaknesses of the proposed approach",
            "trade-offs and disadvantages",
            "future work",
            "challenges and difficulties",
        ]

        for query in gap_queries:
            try:
                chunks = await retrieve_chunks(
                    db=db,
                    query=query,
                    paper_id=paper_id,
                    top_k=5,
                    min_similarity=0.20,
                )
                print(f"\nQuery: '{query}' -> {len(chunks)} chunks")
                for c in chunks:
                    print(f"  Chunk {c['chunk_id']}: final_score={c.get('final_score', 'N/A'):.4f}, similarity={c.get('similarity', 'N/A'):.4f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()

        # Step 2: Full pipeline
        print("\n\n=== STEP 2: FULL PIPELINE ===")
        try:
            result = await detect_research_gaps(
                db=db,
                paper_id=paper_id,
                top_k=5,
            )
            print(f"\nResult paper_id: {result['paper_id']}")
            print(f"Sources count: {len(result['sources'])}")
            print(f"Gaps response:\n{result['gaps']}")
        except Exception as e:
            print(f"FULL PIPELINE ERROR: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
