"""Redis cache abstraction.

Per the architecture (Section 23), the application must still
work if Redis is unavailable. This module provides a transparent
cache layer that gracefully degrades to no-op when Redis is down.
"""

import json
import logging
from typing import Any

from app.core.config import settings


logger = logging.getLogger(__name__)


class RedisCache:
    """Redis-backed cache with graceful degradation.

    When Redis is unavailable, all get/set operations silently
    return defaults — the application continues to function,
    just without caching.
    """

    def __init__(self):
        self._client = None
        self._available = False
        self._connect()

    def _connect(self):
        """Attempt to connect to Redis. Non-fatal on failure."""
        if not settings.REDIS_URL:
            logger.info(
                "Redis URL not configured — running without cache"
            )
            return

        try:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_timeout=3,
                socket_connect_timeout=3,
            )
            self._available = True
            logger.info("Redis cache connected")
        except Exception as exc:
            logger.warning(
                "Redis unavailable, running without cache: %s",
                exc,
            )
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available and self._client is not None

    async def get(self, key: str) -> Any | None:
        """Get a value from cache. Returns None on miss or error."""
        if not self.is_available:
            return None
        try:
            raw = await self._client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.debug("Cache get error for %s: %s", key, exc)
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600,
    ) -> bool:
        """Set a value in cache with TTL (seconds).

        Returns True on success, False on error.
        """
        if not self.is_available:
            return False
        try:
            serialized = json.dumps(value, default=str)
            await self._client.setex(key, ttl, serialized)
            return True
        except Exception as exc:
            logger.debug("Cache set error for %s: %s", key, exc)
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if not self.is_available:
            return False
        try:
            await self._client.delete(key)
            return True
        except Exception as exc:
            logger.debug(
                "Cache delete error for %s: %s", key, exc
            )
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern.

        Returns the number of keys deleted.
        """
        if not self.is_available:
            return 0
        try:
            keys = []
            async for key in self._client.scan_iter(pattern):
                keys.append(key)
            if keys:
                return await self._client.delete(*keys)
            return 0
        except Exception as exc:
            logger.debug(
                "Cache delete_pattern error: %s", exc
            )
            return 0

    async def close(self):
        """Close the Redis connection."""
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

redis_cache = RedisCache()
