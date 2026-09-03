import io

import fitz


MAX_FILE_SIZE = 25 * 1024 * 1024
MAX_PAGES = 200


def validate_pdf(file_bytes: bytes) -> tuple[bool, list[str]]:
    errors = []

    # 1. File size
    if len(file_bytes) > MAX_FILE_SIZE:
        errors.append("PDF file exceeds the 25 MB limit.")

    if not file_bytes:
        errors.append("Uploaded file is empty.")
        return False, errors

    # 2. PDF magic bytes
    if not file_bytes.startswith(b"%PDF-"):
        errors.append("File is not a valid PDF.")

    # Stop before PyMuPDF if basic validation failed
    if errors:
        return False, errors

    # 3. Open PDF and check corruption/encryption/page count
    try:
        document = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )
        try:
            if document.is_encrypted:
                errors.append(
                    "Password-protected or encrypted PDFs are not supported."
                )

            if document.page_count > MAX_PAGES:
                errors.append(
                    "PDF exceeds the 200-page limit."
                )
        finally:
            document.close()

    except Exception:
        errors.append(
            "PDF is corrupted or cannot be opened."
        )

    return len(errors) == 0, errors