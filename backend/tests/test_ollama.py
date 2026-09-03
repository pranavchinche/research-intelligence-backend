import asyncio

from app.services.llm.ollama import generate_answer


async def main():

    prompt = """
You are a research assistant.

Answer the question concisely.

Question:
What is self-attention in the Transformer?
"""

    answer = await generate_answer(prompt)

    print("\nANSWER")
    print("=" * 80)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())