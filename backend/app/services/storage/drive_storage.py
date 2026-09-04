"""Google Drive PDF storage backend.

Uploads PDFs to a configured Google Drive folder and downloads them
to temporary local files for processing.

Requires environment variables:
  GOOGLE_DRIVE_FOLDER_ID
  GOOGLE_CLIENT_ID
  GOOGLE_CLIENT_SECRET
  GOOGLE_REFRESH_TOKEN
"""

import logging
from pathlib import Path

from app.core.config import settings


logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _get_credentials():
    """Build Google OAuth2 credentials from environment variables."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    creds = Credentials(
        token=None,
        refresh_token=settings.GOOGLE_REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.GOOGLE_CLIENT_ID,
        client_secret=settings.GOOGLE_CLIENT_SECRET,
        scopes=_SCOPES,
    )
    creds.refresh(Request())
    return creds


def _get_drive_service():
    """Build a Google Drive API v3 service object."""
    from googleapiclient.discovery import build

    creds = _get_credentials()
    return build("drive", "v3", credentials=creds)


class DriveStorage:
    """Google Drive PDF storage backend."""

    def upload_pdf(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> str:
        """Upload PDF to Google Drive. Returns the Drive file ID."""
        import io
        from googleapiclient.http import MediaIoBaseUpload

        service = _get_drive_service()

        metadata = {
            "name": filename,
            "parents": [settings.GOOGLE_DRIVE_FOLDER_ID],
        }

        media = MediaIoBaseUpload(
            io.BytesIO(file_bytes),
            mimetype="application/pdf",
            resumable=len(file_bytes) > 5 * 1024 * 1024,
        )

        result = (
            service.files()
            .create(
                body=metadata,
                media_body=media,
                fields="id",
            )
            .execute()
        )

        file_id = result["id"]
        logger.info("Uploaded PDF to Drive: %s (id=%s)", filename, file_id)
        return file_id

    def download_to_temp(
        self,
        storage_key: str,
        temp_filename: str,
    ) -> str:
        """Download PDF from Drive to a temporary local file."""
        from googleapiclient.http import MediaIoBaseDownload

        service = _get_drive_service()

        temp_dir = Path(settings.PDF_DOWNLOAD_DIR)
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / temp_filename

        request = service.files().get_media(fileId=storage_key)

        with open(temp_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        logger.info("Downloaded Drive file %s to %s", storage_key, temp_path)
        return str(temp_path)

    def delete(self, storage_key: str) -> bool:
        """Delete a file from Google Drive."""
        try:
            service = _get_drive_service()
            service.files().delete(fileId=storage_key).execute()
            logger.info("Deleted Drive file: %s", storage_key)
            return True
        except Exception as exc:
            logger.warning("Failed to delete Drive file %s: %s", storage_key, exc)
            return False

    def get_metadata(self, storage_key: str) -> dict:
        """Return file metadata from Google Drive."""
        service = _get_drive_service()
        result = (
            service.files()
            .get(
                fileId=storage_key,
                fields="id,name,size,createdTime,mimeType",
            )
            .execute()
        )
        return result
