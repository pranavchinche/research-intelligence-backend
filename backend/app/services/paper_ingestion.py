from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path

from app.services.storage.pdf_downloader import download_pdf
from app.services.validation.normalize import normalize_paper
from app.services.validation.paper_validator import validate_paper
from app.services.dedup.paper_deduplicator import find_duplicate
from app.services.paper_repository import PaperRepository

from app.services.parsing.pdf_parser import extract_pdf_text
from app.services.chunking.chunker import chunk_pages
from app.services.chunking.chunk_repository import create_paper_chunks
from app.services.embedding_pipeline import generate_chunk_embeddings


def _has_value(value) -> bool:
    """Return True for meaningful scalar/list metadata."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(str(item).strip() for item in value)
    return bool(value)


def _apply_pdf_metadata(paper: dict, metadata: dict | None) -> dict:
    """
    Fill missing local-paper metadata from embedded PDF metadata.

    Never overwrites metadata already supplied by arXiv/OpenAlex/etc.
    Never invents an author when the PDF has no embedded author metadata.
    """
    metadata = metadata or {}
    enriched = dict(paper)

    pdf_author = metadata.get("author")
    if not _has_value(enriched.get("authors")) and _has_value(pdf_author):
        enriched["authors"] = [pdf_author.strip()]

    pdf_title = metadata.get("title")
    current_title = enriched.get("title")

    # Local uploads initially use the filename as the title. If the PDF
    # contains a real embedded title, prefer that title.
    if (
        enriched.get("source") == "local"
        and _has_value(pdf_title)
        and (
            not _has_value(current_title)
            or str(current_title).strip() == str(
                enriched.get("source_id", "")
            ).removeprefix("local-").strip()
        )
    ):
        enriched["title"] = pdf_title.strip()

    return enriched


async def ingest_paper(
    db: AsyncSession,
    raw_paper: dict,
    source: str
):
    # ---------------------------------
    # 1. NORMALIZE PAPER
    # ---------------------------------

    paper = normalize_paper(
        raw_paper,
        source
    )

    # ---------------------------------
    # 2. VALIDATE PAPER METADATA
    # ---------------------------------

    is_valid, errors = validate_paper(
        paper
    )

    if not is_valid:
        return {
            "status": "rejected",
            "errors": errors
        }

    # ---------------------------------
    # 3. EARLY DUPLICATE CHECK
    #
    # DOI / arXiv / source ID / title
    # ---------------------------------

    duplicate = await find_duplicate(
        db,
        paper
    )

    if duplicate:
        return {
            "status": "duplicate",
            "paper_id": duplicate.id,
            "title": duplicate.title,
            "reason": "metadata_match"
        }

    # ---------------------------------
    # 4. PREPARE PDF PATH
    # ---------------------------------

    pdf_path = raw_paper.get(
        "pdf_path"
    )

    drive_file_id = raw_paper.get(
        "drive_file_id"
    )

    # If using Google Drive, download to temp for processing
    if not pdf_path and drive_file_id:
        try:
            from app.services.storage.factory import get_storage
            storage = get_storage()
            pdf_path = storage.download_to_temp(
                drive_file_id,
                f"paper_{drive_file_id}.pdf",
            )
        except Exception as e:
            return {
                "status": "rejected",
                "errors": [
                    f"Google Drive download failed: {str(e)}"
                ],
            }

    # ---------------------------------
    # 5. DOWNLOAD PDF IF NECESSARY
    # ---------------------------------

    if not pdf_path and paper.get("pdf_url"):

        try:

            # Temporary filename.
            # We don't have a database ID yet.
            filename = "paper_ingest.pdf"

            pdf_path = await download_pdf(
                paper["pdf_url"],
                filename
            )

        except Exception as e:

            return {
                "status": "rejected",
                "errors": [
                    f"PDF download failed: {str(e)}"
                ]
            }

    # ---------------------------------
    # 6. NO PDF AVAILABLE
    # ---------------------------------

    if not pdf_path:

        # We can still create metadata-only papers.
        paper_data = {
            "title": paper["title"],
            "abstract": paper.get("abstract"),
            "authors": ", ".join(
                paper.get("authors", [])
            ),
            "doi": paper.get("doi"),
            "source": paper["source"],
            "source_id": paper.get("source_id"),
            "arxiv_id": paper.get("arxiv_id"),
            "published_date": paper.get("published_date"),
            "updated_date": paper.get("updated_date"),
            "categories": ", ".join(
                paper.get("categories", [])
            ),
            "pdf_url": paper.get("pdf_url"),
            "pdf_path": None,
            "drive_file_id": drive_file_id,
            "full_text": None,
        }

        saved_paper = await PaperRepository.create(
            db,
            paper_data
        )

        return {
            "status": "created",
            "paper_id": saved_paper.id,
            "title": saved_paper.title,
            "chunks": 0,
            "embedding": 0,
            "message": (
                "Paper created, but no PDF was available."
            )
        }

    # ---------------------------------
    # 7. EXTRACT PDF TEXT + METADATA
    # ---------------------------------

    try:

        parsed = extract_pdf_text(
            pdf_path
        )

    except Exception as e:

        return {
            "status": "rejected",
            "errors": [
                f"PDF extraction failed: {str(e)}"
            ]
        }

    # Embedded PDF metadata is authoritative only for fields that are
    # currently missing from a local upload.
    paper = _apply_pdf_metadata(
        paper,
        parsed.get("metadata")
    )

    # ---------------------------------
    # 8. OCR REQUIRED
    # ---------------------------------

    if parsed.get("needs_ocr"):

        return {
            "status": "rejected",
            "errors": [
                "PDF requires OCR."
            ]
        }

    # ---------------------------------
    # 9. CLEAN FULL TEXT
    # ---------------------------------

    full_text = parsed.get(
        "full_text",
        ""
    ) or ""

    full_text = full_text.replace(
        "\x00",
        ""
    )

    # ---------------------------------
    # 10. CLEAN PAGE TEXT
    # ---------------------------------

    clean_pages = []

    for page in parsed.get(
        "pages",
        []
    ):

        clean_text = page.get(
            "text",
            ""
        ) or ""

        clean_text = clean_text.replace(
            "\x00",
            ""
        )

        clean_page = {
            **page,
            "text": clean_text
        }

        clean_pages.append(
            clean_page
        )

    # ---------------------------------
    # 11. SECOND DUPLICATE CHECK
    #
    # This time full_text is available.
    #
    # This catches exact copies even when
    # filenames/titles are different.
    # ---------------------------------

    paper_with_text = {
        **paper,
        "full_text": full_text,
    }

    duplicate = await find_duplicate(
        db,
        paper_with_text
    )

    if duplicate:

        return {
            "status": "duplicate",
            "paper_id": duplicate.id,
            "title": duplicate.title,
            "reason": "full_text_match"
        }

    # ---------------------------------
    # 12. CREATE PAPER RECORD
    # ---------------------------------

    paper_data = {
        "title": paper["title"],
        "abstract": paper.get("abstract"),
        "authors": ", ".join(
            paper.get("authors", [])
        ),
        "doi": paper.get("doi"),
        "source": paper["source"],
        "source_id": paper.get("source_id"),
        "arxiv_id": paper.get("arxiv_id"),
        "published_date": paper.get("published_date"),
        "updated_date": paper.get("updated_date"),
        "categories": ", ".join(
            paper.get("categories", [])
        ),
        "pdf_url": paper.get("pdf_url"),
        "pdf_path": pdf_path if not drive_file_id else None,
        "drive_file_id": drive_file_id,
        "full_text": full_text,
    }

    saved_paper = await PaperRepository.create(
        db,
        paper_data
    )

    # ---------------------------------
    # 13. CREATE CHUNKS
    # ---------------------------------

    chunks = chunk_pages(
        clean_pages
    )

    chunk_count = await create_paper_chunks(
        db,
        saved_paper.id,
        chunks
    )

    # ---------------------------------
    # 14. GENERATE EMBEDDINGS
    # ---------------------------------

    embedding_count = await generate_chunk_embeddings(
        db,
        saved_paper.id
    )

    # ---------------------------------
    # 15. CLEANUP TEMP PDF
    #
    # When the PDF was downloaded from Google Drive (or a remote URL)
    # into a temporary file, remove it now that processing is done.
    # The PDF is stored permanently in Drive, not in the temp dir.
    # ---------------------------------

    if drive_file_id and pdf_path:
        try:
            cleanup_path = Path(pdf_path)
            if cleanup_path.exists():
                cleanup_path.unlink()
        except Exception:
            pass

    # ---------------------------------
    # 15. FINAL RESPONSE
    # ---------------------------------

    return {
        "status": "created",
        "paper_id": saved_paper.id,
        "title": saved_paper.title,
        "chunks": chunk_count,
        "embedding": embedding_count,
        "pdf_path": pdf_path,
    }
