"""Hash-based duplicate detection.

Exact byte-identical files are caught here via SHA-256 of the
PDF content. This is the highest-confidence dedup signal.
"""

import hashlib


def compute_content_hash(file_bytes: bytes) -> str:
    """Compute SHA-256 hex digest of PDF bytes."""
    return hashlib.sha256(file_bytes).hexdigest()
