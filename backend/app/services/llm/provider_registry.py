"""LLM Provider Registry — tries providers in priority order.

Per the architecture (Section 17), the registry holds an ordered
priority list (config-driven) and on each call tries providers in
order, skipping any that report ``is_available() == False``.  The
provider that actually served the request is recorded for job
tracking.
"""

import logging

from app.core.config import settings
from app.services.llm.base import (
    LLMProvider,
    LLMResponse,
    ProviderError,
)
from app.services.llm.ollama import ollama_provider
from app.services.llm.groq import groq_provider
from app.services.llm.gemini import gemini_provider
from app.services.llm.openrouter import openrouter_provider


logger = logging.getLogger(__name__)


# All known providers, keyed by name
_DEFAULT_PROVIDERS: dict[str, LLMProvider] = {
    "ollama": ollama_provider,
    "groq": groq_provider,
    "gemini": gemini_provider,
    "openrouter": openrouter_provider,
}


class ProviderRegistry:
    """Tries configured providers in priority order with fallback.

    A clean facade for application services: callers only need
    ``registry.generate(prompt)`` — they never learn or care whether
    the request is served by Groq, Gemini, OpenRouter or Ollama.
    """

    def __init__(
        self,
        providers: dict[str, LLMProvider] | None = None,
        chain: list[str] | None = None,
    ):
        # ``dict(providers)`` copies the mapping so tests can build
        # self-contained registries without mutating the singleton.
        self._providers: dict[str, LLMProvider] = dict(
            providers if providers is not None else _DEFAULT_PROVIDERS
        )
        self._chain: list[str] = list(
            chain if chain is not None else settings.llm_fallback_chain
        )
        self._default_provider: str = settings.LLM_DEFAULT_PROVIDER
        self._last_used_provider: str = ""

    # ------------------------------------------------------------------
    # Registration / introspection
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        provider: LLMProvider,
        in_chain: bool = False,
    ) -> None:
        """Register a provider under *name*.

        If ``in_chain`` is true and the name is not already in the
        fallback chain, it is appended at the end of the chain.
        """
        self._providers[name] = provider
        if in_chain and name not in self._chain:
            self._chain.append(name)

    def get(self, name: str) -> LLMProvider | None:
        """Return the registered provider by name, or None."""
        return self._providers.get(name)

    @property
    def providers(self) -> dict[str, LLMProvider]:
        return dict(self._providers)

    @property
    def enabled_providers(self) -> list[str]:
        """Names of every provider currently reporting available."""
        return [
            name
            for name, provider in self._providers.items()
            if provider.is_available()
        ]

    @property
    def chain(self) -> list[str]:
        return list(self._chain)

    @property
    def default_provider(self) -> str:
        return self._default_provider

    @property
    def last_used_provider(self) -> str:
        return self._last_used_provider

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_provider(self) -> str | None:
        """Name of the explicitly configured default provider.

        Returns the default when it is registered and available,
        otherwise None (the caller can fall back to the chain).
        """
        if (
            self._default_provider in self._providers
            and self._providers[self._default_provider].is_available()
        ):
            return self._default_provider
        return None

    def get_provider_status(self) -> dict[str, dict]:
        """Return status of every known provider."""
        result = {}
        for name, provider in self._providers.items():
            result[name] = {
                "available": provider.is_available(),
                "enabled": self._is_enabled(provider),
                "in_chain": name in self._chain,
                "priority": (
                    self._chain.index(name) + 1
                    if name in self._chain
                    else None
                ),
            }
        return result

    def get_active_provider_name(self) -> str | None:
        """Return the name of the first available provider, or None."""
        for name in self._chain:
            provider = self._providers.get(name)
            if provider and provider.is_available():
                return name
        return None

    @staticmethod
    def _is_enabled(provider: LLMProvider) -> bool:
        enabled = getattr(provider, "_enabled", None)
        if enabled is None:
            return True
        return bool(enabled)

    # ------------------------------------------------------------------
    # Generation with fallback
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Try each provider in the fallback chain.

        Returns the first successful response.

        Falls back only on *transient provider failures*
        (:class:`ProviderUnavailableError`) and empty-response
        outcomes.  Auth/config errors for a provider are noted and
        the next provider is still attempted, but the chain never
        retries the same provider.

        Raises :class:`ProviderError` if every provider fails.  A
        retry loop is avoided because:
          - each provider is attempted at most once per call;
          - ``is_available()`` gates disabled/no-key providers;
          - auth/config failures skip the provider.
        """
        _max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        _temperature = temperature or settings.LLM_TEMPERATURE

        last_error: Exception | None = None

        for name in self._chain:
            provider = self._providers.get(name)
            if provider is None:
                logger.warning(
                    "LLM provider %r not found, skipping", name
                )
                continue

            if not provider.is_available():
                logger.info(
                    "LLM provider %r unavailable, skipping", name
                )
                continue

            try:
                logger.info(
                    "Attempting LLM generation with %r", name
                )
                response = await provider.generate(
                    prompt=prompt,
                    max_tokens=_max_tokens,
                    temperature=_temperature,
                    **kwargs,
                )

                if response.text:
                    self._last_used_provider = name
                    logger.info(
                        "LLM provider %r succeeded", name
                    )
                    return response

                logger.warning(
                    "LLM provider %r returned empty response",
                    name,
                )
                last_error = ProviderError(
                    f"Provider {name} returned empty response",
                    provider=name,
                )

            except ProviderError as exc:
                # Auth/config issues won't be fixed by the next
                # provider, but we still record them and continue so
                # a healthy provider downstream can serve the request.
                logger.warning(
                    "LLM provider %r failed: %s", name, exc
                )
                last_error = exc

                if hasattr(provider, "mark_failure"):
                    provider.mark_failure()

            except Exception as exc:  # unexpected programming error
                logger.exception(
                    "LLM provider %r raised an unexpected error", name
                )
                last_error = exc

                if hasattr(provider, "mark_failure"):
                    provider.mark_failure()

        raise ProviderError(
            "All LLM providers failed. "
            f"Last error: {last_error}"
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

provider_registry = ProviderRegistry()


# ---------------------------------------------------------------------------
# Convenience function used by all services
# ---------------------------------------------------------------------------

async def get_llm_answer(
    prompt: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    **kwargs,
) -> str:
    """Generate text using the provider chain.

    Returns the generated text.  Raises :class:`ProviderError` if
    every provider fails or returns empty output.  Callers MUST
    handle the exception — the previous silent-fallback behaviour
    (returning error prose as content) caused downstream services
    to misinterpret failure messages as valid LLM output.
    """
    response = await provider_registry.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    return response.text or ""
