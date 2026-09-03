from sqlalchemy.ext.asyncio import AsyncSession

from app.services.rag.citation_verifier import verify_citations
from app.services.rag.retriever import (
    retrieve_chunks,
    expand_with_neighbors,
    build_context,
)
from app.services.rag.prompt_templates import build_rag_prompt
from app.services.llm import get_llm_answer
from app.core.security.prompt_injection_guard import (
    guard_retrieved_chunks,
)


MIN_SIMILARITY = 0.15


async def ask_rag(
    db: AsyncSession,
    question: str,
    paper_id: int | None = None,
    top_k: int = 8,
    history: list[dict] | None = None,
):

    # --------------------------------------------------------
    # 1. Retrieve semantically relevant chunks
    # --------------------------------------------------------

    results = await retrieve_chunks(
        db=db,
        query=question,
        paper_id=paper_id,
        top_k=top_k,
        min_similarity=MIN_SIMILARITY,
    )

    if not results:
        return {
            "question": question,
            "answer": (
                "The available research context is insufficient "
                "to answer this question."
            ),
            "sources": [],
            "citation_verification": {
                "valid": True,
                "citations": [],
                "invalid_citations": [],
                "missing_citations": False,
            },
        }

    # --------------------------------------------------------
    # 2. Expand around the strongest retrieved chunks
    # --------------------------------------------------------

    expanded_results = await expand_with_neighbors(
        db=db,
        retrieved_chunks=results,
        window=1,
    )

    # --------------------------------------------------------
    # 2b. Apply prompt injection defense
    # --------------------------------------------------------

    expanded_results = guard_retrieved_chunks(expanded_results)

    # --------------------------------------------------------
    # 3. Build grounded context
    # --------------------------------------------------------

    context = build_context(
        expanded_results,
        max_chunks=12,
    )

    # --------------------------------------------------------
    # 4. Generate answer
    # --------------------------------------------------------

    prompt = build_rag_prompt(
        question=question,
        context=context,
        history=history,
    )

    try:
        answer = await get_llm_answer(
            prompt
        )
    except RuntimeError:
        answer = (
            "The available research context is insufficient "
            "to answer this question."
        )

    # --------------------------------------------------------
    # 5. Verify citations against the context actually
    #    supplied to the LLM.
    # --------------------------------------------------------

    citation_check = verify_citations(
        answer=answer,
        source_count=min(
            len(expanded_results),
            12,
        ),
    )

    return {
        "question": question,
        "answer": answer,
        "sources": expanded_results,
        "citation_verification": citation_check,
    }