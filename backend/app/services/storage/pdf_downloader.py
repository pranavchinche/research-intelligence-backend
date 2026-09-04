"""Download a PDF from a remote URL to a temporary local file."""

import httpx
from pathlib import Path

from app.core.config import settings


async def download_pdf(
    pdf_url: str,
    filename: str,
) -> str:
    """Download a PDF from a URL into the temp download directory."""
    if not pdf_url:
        raise ValueError("PDF URL is missing")

    async with httpx.AsyncClient(
        timeout=60,
        follow_redirects=True,
    ) as client:
        response = await client.get(pdf_url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()

    if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
        raise ValueError("URL did not return a PDF")

    temp_dir = Path(settings.PDF_DOWNLOAD_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)

    path = temp_dir / filename
    path.write_bytes(response.content)

    return str(path)
