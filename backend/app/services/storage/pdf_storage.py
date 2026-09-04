"""Save a PDF via the configured storage backend."""

from app.services.storage.factory import get_storage


def save_pdf(
    file_bytes: bytes,
    original_filename: str,
) -> str:
    """Upload PDF to the active storage backend. Returns storage key."""
    storage = get_storage()
    return storage.upload_pdf(file_bytes, original_filename)
