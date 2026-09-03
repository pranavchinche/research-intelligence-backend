# D:\FYP\main\backend\app\services\paper_sources\openalex.py

import httpx


OPENALEX_URL = "https://api.openalex.org/works"


def _reconstruct_abstract(abstract_inverted_index) -> str | None:
    if not abstract_inverted_index:
        return None

    word_positions = []

    for word, positions in abstract_inverted_index.items():
        for position in positions:
            word_positions.append((position, word))

    word_positions.sort()

    return " ".join(
        word for _, word in word_positions
    )


def _extract_categories(work) -> list[str]:
    categories = []

    primary_topic = work.get("primary_topic") or {}

    primary_name = primary_topic.get("display_name")

    if primary_name:
        categories.append(primary_name)

    for topic in work.get("topics", []):
        name = topic.get("display_name")

        if name and name not in categories:
            categories.append(name)

    return categories[:5]


async def search_openalex(
    query: str,
    max_results: int = 5
):
    params = {
        "search": query,
        "per-page": max_results,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                OPENALEX_URL,
                params=params
            )
        response.raise_for_status()
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        raise RuntimeError(
            f"OpenAlex API request failed: {exc}"
        ) from exc

    data = response.json()

    papers = []

    for work in data.get("results", []):

        # -------------------------
        # AUTHORS
        # -------------------------
        authors = []

        for author in work.get("authorships", []):
            author_name = author.get(
                "author", {}
            ).get("display_name")

            if author_name:
                authors.append(author_name)

        # -------------------------
        # PRIMARY LOCATION
        # -------------------------
        primary_location = work.get(
            "primary_location"
        ) or {}

        # Actual PDF URL
        pdf_url = primary_location.get(
            "pdf_url"
        )

        # Landing page URL
        landing_url = primary_location.get(
            "landing_page_url"
        )

        # Fallback PDF URL
        if not pdf_url:
            pdf_url = (
                work.get("best_oa_location", {}) or {}
            ).get("pdf_url")

        # -------------------------
        # PAPER
        # -------------------------
        papers.append(
            {
                "title": work.get("title"),

                "abstract": _reconstruct_abstract(
                    work.get("abstract_inverted_index")
                ),

                "authors": authors,

                "doi": work.get("doi"),

                "source": "openalex",

                "source_id": work.get("id"),

                "arxiv_id": None,

                "published_date": work.get(
                    "publication_date"
                ),

                "updated_date": None,

                "categories": _extract_categories(work),

                "pdf_url": pdf_url,

                "url": landing_url,

                "pdf_path": None,

                "full_text": None,
            }
        )

    return papers