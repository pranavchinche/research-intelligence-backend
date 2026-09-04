"""Abstract storage interface for PDF files.

Production uses Google Drive. Development uses local temporary storage.
"""

from abc import ABC, abstractmethod


class BaseStorage(ABC):
    """Interface that all storage backends must implement."""

    @abstractmethod
    def upload_pdf(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> str:
        """Upload a PDF and return a storage key (e.g. Drive file ID or path)."""

    @abstractmethod
    def download_to_temp(
        self,
        storage_key: str,
        temp_filename: str,
    ) -> str:
        """Download a PDF from storage to a temp file. Return the temp file path."""

    @abstractmethod
    def delete(
        self,
        storage_key: str,
    ) -> bool:
        """Delete a PDF from storage. Return True on success."""

    @abstractmethod
    def get_metadata(
        self,
        storage_key: str,
    ) -> dict:
        """Return file metadata (name, size, created time, etc.)."""
