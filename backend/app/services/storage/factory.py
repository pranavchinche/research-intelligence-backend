"""Storage factory — returns the appropriate backend based on config.

If Google Drive credentials are configured, uses DriveStorage.
Otherwise falls back to local temporary storage.
"""

from app.core.config import settings


def get_storage():
    """Return the active PDF storage backend instance."""
    if settings.use_google_drive:
        from app.services.storage.drive_storage import DriveStorage
        return DriveStorage()
    from app.services.storage.local_storage import LocalStorage
    return LocalStorage()
