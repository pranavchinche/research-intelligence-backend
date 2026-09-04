"""Local temporary PDF storage backend.

Used for development when Google Drive is not configured.
PDFs are stored in a configurable temporary directory.
"""

import uuid
from pathlib import Path

from app.core.config import settings


class LocalStorage:
    """Local filesystem PDF storage backend."""

    def upload_pdf(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> str:
        """Save PDF locally. Returns the file path as the storage key."""
        storage_path = Path(settings.PDF_STORAGE_DIR)
        storage_path.mkdir(parents=True, exist_ok=True)

        file_id = uuid.uuid4().hex
        safe_name = f"{file_id}.pdf"
        file_path = storage_path / safe_name

        file_path.write_bytes(file_bytes)
        return str(file_path)

    def download_to_temp(
        self,
        storage_key: str,
        temp_filename: str,
    ) -> str:
        """For local storage the file is already on disk — return the path."""
        if Path(storage_key).exists():
            return storage_key

        temp_dir = Path(settings.PDF_DOWNLOAD_DIR)
        temp_dir.mkdir(parents=True, exist_ok=True)
        dest = temp_dir / temp_filename

        import shutil
        shutil.copy2(storage_key, dest)
        return str(dest)

    def delete(self, storage_key: str) -> bool:
        """Delete the local PDF file."""
        try:
            path = Path(storage_key)
            if path.exists():
                path.unlink()
            return True
        except Exception:
            return False

    def get_metadata(self, storage_key: str) -> dict:
        """Return basic file metadata."""
        path = Path(storage_key)
        if not path.exists():
            return {"error": "File not found"}
        stat = path.stat()
        return {
            "name": path.name,
            "size": stat.st_size,
            "path": str(path),
        }
