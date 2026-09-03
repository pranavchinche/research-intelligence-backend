# D:\FYP\main\backend\app\api\upload.py

from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.storage.pdf_storage import save_pdf
from app.services.validation.pdf_validator import validate_pdf
from app.services.paper_processing_service import process_paper


router = APIRouter(
    prefix="/upload",
    tags=["Upload"],
)


@router.post("/pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    # -------------------------
    # Basic filename validation
    # -------------------------

    filename = file.filename or ""

    if not filename.lower().endswith(".pdf"):
        return {
            "status": "rejected",
            "filename": filename,
            "errors": [
                "Only PDF files are supported."
            ],
        }

    # -------------------------
    # Read file
    # -------------------------

    file_bytes = await file.read()

    # -------------------------
    # Validate PDF
    # -------------------------

    is_valid, errors = validate_pdf(
        file_bytes
    )

    if not is_valid:
        return {
            "status": "rejected",
            "filename": filename,
            "errors": errors,
        }

    # -------------------------
    # Save PDF
    # -------------------------

    pdf_path = save_pdf(
        file_bytes=file_bytes,
        original_filename=filename,
    )

    # -------------------------
    # Build local paper
    # -------------------------

    raw_paper = {
        "title": filename,
        "abstract": "",
        "authors": [],
        "doi": None,
        "source": "local",
        "source_id": f"local-{filename}",
        "arxiv_id": None,
        "published_date": None,
        "updated_date": None,
        "categories": [],
        "pdf_url": None,
        "pdf_path": pdf_path,
    }

    # -------------------------
    # Ingest and process
    # -------------------------

    try:

        result = await process_paper(
            db=db,
            raw_paper=raw_paper,
            source="local",
        )

    except Exception as exc:

        # If ingestion crashes after the file was saved,
        # remove the unused PDF from storage.

        try:
            Path(pdf_path).unlink(
                missing_ok=True
            )
        except Exception:
            pass

        return {
            "status": "error",
            "filename": filename,
            "pdf_path": pdf_path,
            "message": str(exc),
        }

    # -------------------------
    # Duplicate handling
    # -------------------------

    if result.get("status") == "duplicate":

        # The uploaded PDF is not needed because
        # an existing paper already represents it.

        try:
            Path(pdf_path).unlink(
                missing_ok=True
            )
        except Exception:
            pass

        return {
            "status": "duplicate",
            "filename": filename,
            "pdf_path": None,
            "paper_id": result.get("paper_id"),
            "title": result.get("title"),
            "chunks": 0,
            "embedding": 0,
            "message": result.get("message"),
        }

    # -------------------------
    # Return result
    # -------------------------

    return {
        "status": result.get(
            "status",
            "created"
        ),
        "filename": filename,
        "pdf_path": pdf_path,
        "paper_id": result.get(
            "paper_id"
        ),
        "title": result.get(
            "title"
        ),
        "chunks": result.get(
            "chunks",
            0
        ),
        "embedding": result.get(
            "embedding",
            0
        ),
        "message": result.get(
            "message"
        ),
    }