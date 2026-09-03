import asyncio

from app.core.database import AsyncSessionlocal
from app.services.rag.rag_service import ask_rag


TEST_CASES = [
    {
        "question": "What optimizer was used to train the Transformer?",
        "keywords": ["Adam", "0.9", "0.98"]
    },
    {
        "question": "What learning rate schedule was used?",
        "keywords": ["learning rate", "warmup", "4000"]
    },
    {
        "question": "How many layers does the Transformer encoder have?",
        "keywords": ["6"]
    },
    {
        "question": "What is the model dimension?",
        "keywords": ["512"]
    },
    {
        "question": "How many attention heads were used?",
        "keywords": ["8"]
    },
    {
        "question": "What datasets were used to train the Transformer?",
        "keywords": ["WMT 2014", "English-German", "English-French"]
    },
    {
        "question": "What dropout rate was used for the base model?",
        "keywords": ["0.1"]
    },
    {
        "question": "How long was the base model trained?",
        "keywords": ["100,000", "12 hours"]
    },
    {
        "question": "What is self-attention?",
        "keywords": ["self-attention"]
    },
    {
        "question": "What is the weather in Nagpur today?",
        "keywords": []
    }
]


async def main():

    passed = 0
    evaluated = 0

    print("\n" + "=" * 70)
    print("RAG EVALUATION")
    print("=" * 70)

    async with AsyncSessionlocal() as db:

        for i, case in enumerate(TEST_CASES, 1):

            question = case["question"]
            keywords = case["keywords"]

            print(f"\nTEST {i}")
            print("-" * 70)
            print("QUESTION:", question)

            try:

                result = await ask_rag(
                    db=db,
                    question=question,
                    top_k=20,
                )

                answer = result["answer"]

                print("\nANSWER:")
                print(answer)

                if not keywords:

                    print("\nSTATUS: MANUAL CHECK")
                    continue

                evaluated += 1

                missing = [
                    keyword
                    for keyword in keywords
                    if keyword.lower() not in answer.lower()
                ]

                if not missing:

                    print("\nSTATUS: PASS")
                    passed += 1

                else:

                    print("\nSTATUS: FAIL")
                    print("Missing:", missing)

            except Exception as e:

                print("\nSTATUS: ERROR")
                print(type(e).__name__, e)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Passed: {passed}/{evaluated}")

    if evaluated:
        print(f"Score: {(passed / evaluated) * 100:.1f}%")
    else:
        print("Score: N/A")


if __name__ == "__main__":
    asyncio.run(main())