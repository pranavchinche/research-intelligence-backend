"""Diagnostic: examine actual chunk texts used as sources."""
import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app.core.database import AsyncSessionlocal
from app.services.rag.retriever import retrieve_chunks, expand_with_neighbors, build_context

async def main():
    paper_id = 1
    async with AsyncSessionlocal() as db:
        # Replicate the pipeline's chunk selection
        all_results = {}
        gap_queries = [
            "limitations of the proposed method", "limitations of the model",
            "limitations of the experiments", "weaknesses of the proposed approach",
            "problems with the proposed approach", "performance degradation",
            "trade-offs and disadvantages", "future work", "future improvements",
            "unresolved problems", "challenges and difficulties", "what could be improved",
            "inability of the proposed method", "does not perform well",
            "remains difficult", "does not have a complete explanation",
        ]
        for query in gap_queries:
            retrieved = await retrieve_chunks(db=db, query=query, paper_id=paper_id, top_k=5, min_similarity=0.20)
            for r in (retrieved or []):
                if not isinstance(r, dict) or "paper_id" not in r:
                    continue
                key = (r["paper_id"], r["chunk_id"])
                if key not in all_results:
                    all_results[key] = r
                else:
                    old = all_results[key].get("final_score", all_results[key].get("similarity", 0.0) or 0.0)
                    new = r.get("final_score", r.get("similarity", 0.0) or 0.0)
                    if new > old:
                        all_results[key] = r

        results = list(all_results.values())
        # Filter
        results = [r for r in results if r.get("paper_id") == paper_id]
        results = [r for r in results if (r.get("final_score") or 0.0) >= 0.18]

        import re
        _REFERENCE_HEADINGS = ("references", "bibliography", "acknowledgments", "acknowledgements", "author contributions", "conflict of interest", "conflicts of interest")
        def _is_reference_chunk(text):
            if not text: return False
            lower = text.strip().lower()
            first_line = lower.splitlines()[0].strip()
            if any(first_line.startswith(h) or first_line == h for h in _REFERENCE_HEADINGS): return True
            markers = re.findall(r"\[\d{1,3}\]", text)
            if len(markers) >= 5: return True
            return False

        results = [r for r in results if not _is_reference_chunk(r.get("text", ""))]
        results.sort(key=lambda x: x.get("final_score", x.get("similarity", 0.0) or 0.0), reverse=True)
        candidate_results = results[:5]

        expanded = await expand_with_neighbors(db=db, retrieved_chunks=candidate_results, window=1)
        # Dedup
        unique = {}
        for c in expanded:
            if isinstance(c, dict) and "paper_id" in c and "chunk_id" in c:
                key = (c["paper_id"], c["chunk_id"])
                if key not in unique: unique[key] = c
        expanded = list(unique.values())
        expanded = [c for c in expanded if c.get("paper_id") == paper_id]
        expanded = [c for c in expanded if not _is_reference_chunk(c.get("text", ""))]
        expanded.sort(key=lambda x: x.get("final_score", x.get("similarity", 0.0) or 0.0), reverse=True)
        expanded = expanded[:15]

        # Print numbered sources
        for i, chunk in enumerate(expanded, 1):
            ct = "NEIGHBOR" if chunk.get("is_neighbor") else "RETRIEVED"
            text = chunk.get("text", "")
            print(f"\n{'='*60}")
            print(f"SOURCE {i} | Chunk {chunk['chunk_id']} | {ct}")
            print(f"Text ({len(text)} chars):")
            print(text[:500])
            if len(text) > 500:
                print(f"... ({len(text) - 500} more chars)")

asyncio.run(main())
