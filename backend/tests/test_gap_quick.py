"""Quick test for paper 1 with updated prompt."""
import asyncio, json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from app.core.database import AsyncSessionlocal
from app.services.gaps.gap_pipeline import detect_research_gaps

async def main():
    paper_id = 1
    async with AsyncSessionlocal() as db:
        result = await detect_research_gaps(db=db, paper_id=paper_id, top_k=5)
        gaps = json.loads(result["gaps"])
        print(f"\nPaper {paper_id}:")
        print(f"  Limitations: {len(gaps['limitations'])}")
        for lim in gaps["limitations"]:
            print(f"    - [{lim['status']}] {lim['statement']}")
        print(f"  Possible gaps: {len(gaps['possible_gaps'])}")
        for gap in gaps["possible_gaps"]:
            print(f"    - {gap['statement']}")

asyncio.run(main())
