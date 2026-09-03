"""Retry policy for the async processing pipeline.

Per the architecture (Section 21):
- Transient failures → retry with exponential backoff
- Permanent failures → fail immediately
- No infinite retry loops (max_retries cap)
"""

import math
from datetime import datetime, timedelta, timezone


# Default retry configuration
DEFAULT_MAX_RETRIES = 3
BASE_DELAY_SECONDS = 5
MAX_DELAY_SECONDS = 300  # 5 minutes

# Error categories
RETRYABLE_ERRORS = {
    "timeout",
    "rate_limit",
    "network_error",
    "temporary",
    "connection",
    "503",
    "429",
}

PERMANENT_ERRORS = {
    "validation",
    "corrupted",
    "not_found",
    "permission",
    "unauthorized",
}


def is_retryable(error_message: str) -> bool:
    """Determine if an error is transient and worth retrying.

    Returns True for network/timeout/rate-limit errors.
    Returns False for permanent failures (validation, corrupted, etc.).
    """
    if not error_message:
        return True

    lower = error_message.lower()

    for permanent in PERMANENT_ERRORS:
        if permanent in lower:
            return False

    for retryable in RETRYABLE_ERRORS:
        if retryable in lower:
            return True

    return True


def calculate_next_retry(
    retry_count: int,
    base_delay: float = BASE_DELAY_SECONDS,
    max_delay: float = MAX_DELAY_SECONDS,
) -> datetime:
    """Calculate the timestamp for the next retry attempt.

    Uses exponential backoff: delay = base_delay * 2^retry_count,
    capped at max_delay.
    """
    delay = min(
        base_delay * math.pow(2, retry_count),
        max_delay,
    )
    return datetime.now(timezone.utc) + timedelta(seconds=delay)


def can_retry(
    retry_count: int,
    max_retries: int = DEFAULT_MAX_RETRIES,
    error_message: str = "",
) -> bool:
    """Check if a job should be retried.

    Returns True only if:
    1. We haven't exhausted retries
    2. The error is classified as retryable
    """
    if retry_count >= max_retries:
        return False
    return is_retryable(error_message)


def get_retry_info(
    retry_count: int,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    """Return retry status information for a job."""
    can_attempt = retry_count < max_retries
    next_delay = min(
        BASE_DELAY_SECONDS * math.pow(2, retry_count),
        MAX_DELAY_SECONDS,
    )
    return {
        "retry_count": retry_count,
        "max_retries": max_retries,
        "can_retry": can_attempt,
        "next_delay_seconds": next_delay if can_attempt else None,
        "next_retry_at": (
            calculate_next_retry(retry_count)
            if can_attempt
            else None
        ),
    }
