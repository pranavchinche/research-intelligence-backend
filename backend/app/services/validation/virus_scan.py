"""Virus scanning for uploaded PDFs.

Per the architecture (Section 19), a lightweight virus scan is
run before files touch persistent storage.

If ClamAV is not installed, the scan is skipped gracefully —
the file is NOT labeled as "virus-free", it is labeled as
"scan_not_available".
"""

import shutil
import subprocess
import logging


logger = logging.getLogger(__name__)


def _clamav_available() -> bool:
    """Check if ClamAV (clamscan) is installed."""
    return shutil.which("clamscan") is not None


def scan_file(file_path: str) -> dict:
    """Scan a file for viruses using ClamAV.

    Returns:
        {
            "status": "clean" | "infected" | "scan_not_available" | "error",
            "details": str,
            "engine": "clamav" | "none",
        }
    """
    if not _clamav_available():
        return {
            "status": "scan_not_available",
            "details": (
                "ClamAV is not installed. "
                "File was not virus-scanned."
            ),
            "engine": "none",
        }

    try:
        result = subprocess.run(
            ["clamscan", "--no-summary", file_path],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            return {
                "status": "clean",
                "details": "No threats detected.",
                "engine": "clamav",
            }
        elif result.returncode == 1:
            return {
                "status": "infected",
                "details": result.stdout[:500],
                "engine": "clamav",
            }
        else:
            return {
                "status": "error",
                "details": result.stderr[:500],
                "engine": "clamav",
            }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "details": "Virus scan timed out after 60s.",
            "engine": "clamav",
        }
    except Exception as exc:
        return {
            "status": "error",
            "details": f"Scan failed: {str(exc)[:200]}",
            "engine": "clamav",
        }
