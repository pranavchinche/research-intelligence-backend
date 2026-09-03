"""Connectivity Guard — determines system operating mode.

Per the architecture (Section 16), the system has three modes:
- online: all services reachable
- degraded: some services unreachable
- offline: only local DB/vector DB reachable

The guard caches the combined mode so it doesn't need to ping
every service on every request.
"""

import time
import logging

from app.services.connectivity.health_checker import health_checker


logger = logging.getLogger(__name__)


# System modes
MODE_ONLINE = "online"
MODE_DEGRADED = "degraded"
MODE_OFFLINE = "offline"


class ConnectivityGuard:
    """Determines and caches the system operating mode."""

    def __init__(self):
        self._mode: str = MODE_ONLINE
        self._last_evaluated: float = 0.0
        self._evaluate_interval: int = 30  # seconds

    @property
    def mode(self) -> str:
        return self._mode

    async def evaluate(self, force: bool = False) -> str:
        """Re-evaluate system mode if interval has elapsed.

        Returns the current mode string.
        """
        now = time.time()
        if not force and (now - self._last_evaluated) < self._evaluate_interval:
            return self._mode

        statuses = await health_checker.check_all()
        self._last_evaluated = now

        reachable_count = sum(
            1 for s in statuses.values() if s["reachable"]
        )
        total = len(statuses)

        if reachable_count == total:
            self._mode = MODE_ONLINE
        elif reachable_count > 0:
            self._mode = MODE_DEGRADED
        else:
            self._mode = MODE_OFFLINE

        logger.info(
            "System mode: %s (%d/%d services reachable)",
            self._mode,
            reachable_count,
            total,
        )
        return self._mode

    def get_mode_sync(self) -> str:
        """Return last-known mode without network calls."""
        return self._mode

    def get_status_detail(self) -> dict:
        """Return detailed status for the /system/mode endpoint."""
        service_statuses = health_checker.get_cached_status()
        llm_providers = {}
        try:
            from app.services.llm.provider_registry import provider_registry
            llm_providers = provider_registry.get_provider_status()
        except Exception:
            pass

        return {
            "mode": self._mode,
            "services": service_statuses,
            "llm_providers": llm_providers,
        }


connectivity_guard = ConnectivityGuard()
