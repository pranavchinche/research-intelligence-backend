"""Health checker for external dependencies.

Pings each external service on a configurable interval and
caches the result so checks are not performed per-request.
"""

import time
import logging

import httpx

from app.core.config import settings


logger = logging.getLogger(__name__)


class ServiceHealth:
    """Tracks health state for a single external service."""

    def __init__(self, name: str, url: str, timeout: float = 5.0):
        self.name = name
        self.url = url
        self.timeout = timeout
        self.reachable: bool = False
        self.last_checked: float = 0.0
        self.last_error: str = ""

    def is_stale(self, interval: float) -> bool:
        return (time.time() - self.last_checked) > interval

    async def check(self) -> bool:
        """Ping the service. Returns True if reachable."""
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout
            ) as client:
                response = await client.get(self.url)
                self.reachable = response.status_code < 500
                self.last_error = ""
        except Exception as exc:
            self.reachable = False
            self.last_error = str(exc)[:200]

        self.last_checked = time.time()
        return self.reachable


class HealthChecker:
    """Checks all external dependencies and caches results."""

    def __init__(self):
        self._interval = settings.CONNECTIVITY_CHECK_INTERVAL
        self._timeout = settings.CONNECTIVITY_TIMEOUT

        self.services: dict[str, ServiceHealth] = {
            "arxiv": ServiceHealth(
                "arxiv",
                "http://export.arxiv.org/api/query?search_query=test&max_results=1",
                self._timeout,
            ),
            "openalex": ServiceHealth(
                "openalex",
                "https://api.openalex.org/works?per-page=1",
                self._timeout,
            ),
            "ollama": ServiceHealth(
                "ollama",
                f"{settings.OLLAMA_BASE_URL}/api/tags",
                self._timeout,
            ),
        }

    async def check_all(self) -> dict[str, dict]:
        """Check all services. Returns status dict."""
        results = {}
        for name, service in self.services.items():
            if service.is_stale(self._interval):
                await service.check()
            results[name] = {
                "reachable": service.reachable,
                "last_error": service.last_error,
                "last_checked": service.last_checked,
            }
        return results

    async def check_service(self, name: str) -> dict:
        """Check a single service by name."""
        service = self.services.get(name)
        if service is None:
            return {"reachable": False, "last_error": "Unknown service"}
        if service.is_stale(self._interval):
            await service.check()
        return {
            "reachable": service.reachable,
            "last_error": service.last_error,
        }

    def get_cached_status(self) -> dict[str, dict]:
        """Return last-known status without making network calls."""
        results = {}
        for name, service in self.services.items():
            results[name] = {
                "reachable": service.reachable,
                "last_error": service.last_error,
            }
        return results


health_checker = HealthChecker()
