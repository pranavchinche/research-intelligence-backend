import asyncio

from app.core.database import AsyncSessionlocal

from app.services.rag.retriever import (
    retrieve_chunks,
    build_context,
)

from app.services.rag.prompt_templates import build_rag_prompt

from app.services.llm.ollama import generate_answer


async def main():

    async with AsyncSessionlocal() as db:

        question = "What optimizer and learning rate were used to train the Transformer?"

        # 1. Retrieve relevant chunks
        results = await retrieve_chunks(
            db=db,
            query=question,
            top_k=10
        )

        # 2. Build research context
        context = build_context(results)

        # 3. Build grounded RAG prompt
        prompt = build_rag_prompt(
            question=question,
            context=context
        )

        print("\nRAG PROMPT")
        print("=" * 80)
        print(prompt)

        # 4. Send RAG prompt to Ollama
        answer = await generate_answer(prompt)

        # 5. Display final answer
        print("\n\nFINAL ANSWER")
        print("=" * 80)
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())