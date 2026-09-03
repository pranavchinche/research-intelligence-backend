import httpx
import xml.etree.ElementTree as ET

ARXIV_URL = "https://export.arxiv.org/api/query"

NAMESPACE = {
    "atom": "http://www.w3.org/2005/Atom"
}


async def search_arxiv(query: str, max_results: int = 5):

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                ARXIV_URL,
                params=params
            )
        response.raise_for_status()
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        raise RuntimeError(
            f"arXiv API request failed: {exc}"
        ) from exc

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as exc:
        raise RuntimeError(
            f"arXiv returned invalid XML: {exc}"
        ) from exc

    papers = []

    for entry in root.findall("atom:entry", NAMESPACE):

        title = entry.findtext(
            "atom:title",
            default="",
            namespaces=NAMESPACE
        ).strip()

        summary = entry.findtext(
            "atom:summary",
            default="",
            namespaces=NAMESPACE
        ).strip()

        published = entry.findtext(
            "atom:published",
            default="",
            namespaces=NAMESPACE
        )

        paper_id = entry.findtext(
            "atom:id",
            default="",
            namespaces=NAMESPACE
        )

        # Authors
        authors = []

        for author in entry.findall("atom:author", NAMESPACE):
            name = author.findtext(
                "atom:name",
                default="",
                namespaces=NAMESPACE
            ).strip()

            if name:
                authors.append(name)

        # Categories
        categories = []

        for category in entry.findall("atom:category", NAMESPACE):
            term = category.get("term")

            if term:
                categories.append(term)

        # ArXiv ID
        arxiv_id = paper_id.split("/abs/")[-1] if "/abs/" in paper_id else None

        # PDF URL
        pdf_url = (
            paper_id.replace("/abs/", "/pdf/") + ".pdf"
            if paper_id
            else None
        )

        papers.append(
            {
                "title": title,
                "abstract": summary,
                "authors": authors,
                "doi": None,
                "source": "arxiv",
                "source_id": paper_id,
                "arxiv_id": arxiv_id,
                "published_date": published,
                "updated_date": None,
                "categories": categories,
                "pdf_url": pdf_url,
                "pdf_path": None,
                "full_text": None,
            }
        )

    return papers