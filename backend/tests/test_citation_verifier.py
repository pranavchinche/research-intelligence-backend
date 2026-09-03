from app.services.rag.citation_verifier import (
    extract_citations,
    verify_citations,
)


def test():

    answer = (
    "Self-attention processes relationships between tokens "
    "[SOURCE 1] and uses self-attention layers [SOURCE 7]."
    )

    citations = extract_citations(answer)

    print("\nEXTRACTED CITATIONS")
    print("=" * 80)
    print(citations)

    result = verify_citations(
        answer=answer,
        source_count=3,
    )

    print("\nVERIFICATION")
    print("=" * 80)
    print(result)


if __name__ == "__main__":
    test()