"""Groq LLM provider — fast inference via Groq Cloud free tier."""

import httpx

from app.core.config import settings
from app.services.llm.base import (
    LLMProvider,
    LLMResponse,
    ProviderAuthError,
    ProviderConfigError,
    ProviderUnavailableError,
)


class GroqProvider(LLMProvider):
    """Groq Cloud provider (Llama-3.1 via Groq)."""

    name = "groq"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        # Allow tests to inject config without touching the global
        # settings singleton.
        self._api_key = (
            api_key if api_key is not None else settings.GROQ_API_KEY
        )
        self._model = model or settings.GROQ_MODEL
        self._base_url = base_url or settings.GROQ_BASE_URL
        self._enabled = settings.GROQ_ENABLED
        self._timeout = (
            getattr(settings, "GROQ_TIMEOUT", 0.0)
            or settings.LLM_TIMEOUT
        )
        self._available: bool | None = None
        self._consecutive_failures = 0

    def is_available(self) -> bool:
        if not self._enabled:
            return False
        if not self._api_key:
            return False
        if self._consecutive_failures >= 3:
            return False
        return True

    def mark_failure(self):
        self._consecutive_failures += 1

    def mark_success(self):
        self._consecutive_failures = 0

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        **kwargs,
    ) -> LLMResponse:
        if not self._enabled:
            raise ProviderConfigError(
                "Groq provider is disabled.",
                provider=self.name,
            )
        if not self._api_key:
            raise ProviderConfigError(
                "Groq API key is not configured.",
                provider=self.name,
            )

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        timeout = kwargs.get("timeout", self._timeout)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    url, json=payload, headers=headers,
                )
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(
                f"Groq request timed out: {exc}",
                provider=self.name,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"Groq network error: {exc}",
                provider=self.name,
            ) from exc

        if response.status_code == 401 or response.status_code == 403:
            raise ProviderAuthError(
                "Groq authentication failed (invalid API key).",
                provider=self.name,
            )
        if response.status_code >= 500:
            raise ProviderUnavailableError(
                f"Groq returned HTTP {response.status_code}",
                provider=self.name,
            )
        if response.status_code == 429:
            raise ProviderUnavailableError(
                "Groq rate limit exceeded.",
                provider=self.name,
            )
        if response.status_code != 200:
            raise ProviderUnavailableError(
                f"Groq returned HTTP {response.status_code}",
                provider=self.name,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderUnavailableError(
                "Groq returned a malformed response",
                provider=self.name,
            ) from exc

        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        self.mark_success()

        return LLMResponse(
            text=text,
            provider=self.name,
            model=self._model,
            usage=data.get("usage", {}),
            raw=data,
        )


groq_provider = GroqProvider()
