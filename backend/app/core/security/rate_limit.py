"""Rate limiting middleware.

Per the architecture (Section 24), all AI-heavy endpoints are
rate-limited per user/IP to protect shared free-tier quotas.

Uses an in-memory token bucket when Redis is unavailable.
"""

import time
import logging
from collections import defaultdict

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings


logger = logging.getLogger(__name__)


class TokenBucket:
    """Simple in-memory token bucket rate limiter."""

    def __init__(
        self,
        max_tokens: int = 30,
        refill_interval: int = 60,
    ):
        self.max_tokens = max_tokens
        self.refill_interval = refill_interval
        self._buckets: dict[str, dict] = defaultdict(
            lambda: {
                "tokens": max_tokens,
                "last_refill": time.time(),
            }
        )

    def allow(self, key: str) -> bool:
        """Check if a request is allowed under the rate limit."""
        bucket = self._buckets[key]
        now = time.time()

        elapsed = now - bucket["last_refill"]
        if elapsed >= self.refill_interval:
            bucket["tokens"] = self.max_tokens
            bucket["last_refill"] = now

        if bucket["tokens"] > 0:
            bucket["tokens"] -= 1
            return True
        return False

    def remaining(self, key: str) -> int:
        bucket = self._buckets[key]
        now = time.time()
        elapsed = now - bucket["last_refill"]
        if elapsed >= self.refill_interval:
            return self.max_tokens
        return max(0, bucket["tokens"])


# Global rate limiter instance
_rate_limiter = TokenBucket(
    max_tokens=settings.RATE_LIMIT_REQUESTS,
    refill_interval=settings.RATE_LIMIT_WINDOW,
)


def get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Starlette middleware for rate limiting."""

    # Endpoints exempt from rate limiting
    EXEMPT_PATHS = {
        "/health",
        "/",
        "/docs",
        "/openapi.json",
        "/redoc",
    }

    # Endpoints with stricter limits (AI-heavy)
    STRICT_PATHS = {
        "/chat",
        "/summary",
        "/gaps",
        "/strengths",
        "/novelty",
        "/compare",
    }

    async def dispatch(self, request: Request, call_next):
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)

        path = request.url.path.rstrip("/")

        # Skip rate limiting for exempt paths
        if path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Skip for non-AI endpoints (papers, search, upload, etc.)
        is_strict = any(
            path.startswith(strict) for strict in self.STRICT_PATHS
        )

        if not is_strict:
            return await call_next(request)

        client_ip = get_client_ip(request)
        rate_key = f"rate:{client_ip}"

        if not _rate_limiter.allow(rate_key):
            remaining = _rate_limiter.remaining(rate_key)
            logger.warning(
                "Rate limit exceeded for %s", client_ip
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "X-RateLimit-Limit": str(
                        settings.RATE_LIMIT_REQUESTS
                    ),
                    "X-RateLimit-Remaining": str(remaining),
                    "Retry-After": str(
                        settings.RATE_LIMIT_WINDOW
                    ),
                },
            )

        response = await call_next(request)

        remaining = _rate_limiter.remaining(rate_key)
        response.headers["X-RateLimit-Limit"] = str(
            settings.RATE_LIMIT_REQUESTS
        )
        response.headers["X-RateLimit-Remaining"] = str(
            remaining
        )

        return response
