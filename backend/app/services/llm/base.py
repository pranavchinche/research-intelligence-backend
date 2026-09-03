"""Abstract base class for LLM providers.

Every provider must implement ``generate`` and ``is_available``.
The ``name`` attribute is used in logging and job tracking.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ProviderError(RuntimeError):
    """Base class for LLM provider errors.

    Raised when a provider cannot serve a request.  Subclasses
    distinguish transient (retryable) from permanent failures so
    the registry can decide whether to fall back to the next
    provider or stop.

    Inherits from :class:`RuntimeError` so existing application
    services that guard LLM calls with ``except RuntimeError``
    keep working unchanged.
    """

    def __init__(self, message: str, provider: str = ""):
        super().__init__(message)
        self.provider = provider


class ProviderUnavailableError(ProviderError):
    """The provider could not be reached / is down (transient).

    Network failures, timeouts, HTTP 5xx, capacity/rate-limit
    errors are all treated as transient: the registry SHOULD fall
    back to the next provider.
    """


class ProviderAuthError(ProviderError):
    """Authentication/authorization failed (permanent).

    Missing or invalid API key.  Falling back will not help; the
    registry SHOULD NOT retry this provider, but MAY continue to
    the next one.
    """


class ProviderConfigError(ProviderError):
    """The provider is misconfigured (disabled, missing key)."""


@dataclass
class LLMResponse:
    """Standardised response from any LLM provider."""

    text: str
    provider: str
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


class LLMProvider(ABC):
    """Abstract interface that every LLM provider must satisfy."""

    name: str = "base"

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        **kwargs,
    ) -> LLMResponse:
        """Send *prompt* to the LLM and return the response.

        Error contract (used by the registry to decide on fallback):

        - Raise :class:`ProviderUnavailableError` for transient
          failures (network, timeout, HTTP 5xx, rate limit).  The
          registry MAY fall back to the next provider.
        - Raise :class:`ProviderAuthError` for auth failures
          (invalid API key).  The registry should not keep retrying
          this provider within the same request.
        - Raise :class:`ProviderConfigError` when the provider is
          disabled/misconfigured (e.g. missing API key).
        - ``LLMResponse`` with empty ``text`` indicates a
          permanent response-time failure (malformed body / empty
          completion) that should not be retried.

        Implementations MUST NOT raise the raw ``httpx``/socket
        exceptions up to application callers.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Cheap check: can this provider handle a request right now?

        A ``True`` return does **not** guarantee success (network
        errors can still happen), but ``False`` means the provider
        should be skipped entirely (e.g. disabled, no API key
        configured, quota exhausted on a previous call).
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
