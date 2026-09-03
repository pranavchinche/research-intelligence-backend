"""Diagnostic: trace the LLM response and validation for paper 13."""
import asyncio
import json
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app.core.database import AsyncSessionlocal
from app.services.gaps.gap_pipeline import validate_gap_answer, build_gap_prompt
from app.services.rag.retriever import retrieve_chunks, expand_with_neighbors, build_context
from app.services.llm.ollama import generate_answer
import re

_REFERENCE_HEADINGS = ("references", "bibliography", "acknowledgments", "acknowledgements", "author contributions", "conflict of interest", "conflicts of interest")

def _is_reference_chunk(text):
    if not text: return False
    lower = text.strip().lower()
    first_line = lower.splitlines()[0].strip()
    if any(first_line.startswith(h) or first_line == h for h in _REFERENCE_HEADINGS): return True
    markers = re.findall(r"\[\d{1,3}\]", text)
    if len(markers) >= 5: return True
    words = text.split()
    if words and (len(markers) / max(len(words), 1)) > 0.05 and len(markers) >= 3: return True
    return False

async def main():
    paper_id = 13
    async with AsyncSessionlocal() as db:
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
                if not isinstance(r, dict) or "paper_id" not in r: continue
                key = (r["paper_id"], r["chunk_id"])
                if key not in all_results: all_results[key] = r
                else:
                    old = all_results[key].get("final_score", all_results[key].get("similarity", 0.0) or 0.0)
                    new = r.get("final_score", r.get("similarity", 0.0) or 0.0)
                    if new > old: all_results[key] = r

        results = list(all_results.values())
        results = [r for r in results if r.get("paper_id") == paper_id]
        results = [r for r in results if not _is_reference_chunk(r.get("text", ""))]
        results = [r for r in results if (r.get("final_score") or 0.0) >= 0.18]
        results.sort(key=lambda x: x.get("final_score", x.get("similarity", 0.0) or 0.0), reverse=True)
        
        candidate_results = results[:5]
        print(f"Candidates: {len(candidate_results)}")
        for c in candidate_results:
            print(f"  Chunk {c['chunk_id']}: score={c.get('final_score', 0):.4f}")
            print(f"    Text[:200]: {c.get('text', '')[:200]}")

        expanded = await expand_with_neighbors(db=db, retrieved_chunks=candidate_results, window=1)
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
        
        print(f"\nExpanded context chunks: {len(expanded)}")
        for i, c in enumerate(expanded, 1):
            ct = "NEIGHBOR" if c.get("is_neighbor") else "RETRIEVED"
            print(f"  SOURCE {i} | Chunk {c['chunk_id']} | {ct}")
            print(f"    Text[:150]: {c.get('text', '')[:150]}")

        context = build_context(expanded, max_chunks=15)
        prompt = build_gap_prompt(context)
        
        print(f"\nPrompt length: {len(prompt)} chars")
        print("\nCalling LLM...")
        raw_answer = await generate_answer(prompt)
        print(f"\nRAW LLM ANSWER:\n{raw_answer}")
        
        print("\n\nValidating...")
        answer = validate_gap_answer(raw_answer, source_chunks=expanded, requested_paper_id=paper_id)
        print(f"\nVALIDATED ANSWER:\n{answer}")

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
asyncio.run(main())
