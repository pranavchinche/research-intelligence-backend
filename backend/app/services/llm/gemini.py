"""Gemini LLM provider — Google AI Studio free tier."""

import httpx

from app.core.config import settings
from app.services.llm.base import (
    LLMProvider,
    LLMResponse,
    ProviderAuthError,
    ProviderConfigError,
    ProviderUnavailableError,
)


class GeminiProvider(LLMProvider):
    """Google Gemini (AI Studio) provider."""

    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        # Allow tests to inject config without touching the global
        # settings singleton.
        self._api_key = (
            api_key if api_key is not None else settings.GEMINI_API_KEY
        )
        self._model = model or settings.GEMINI_MODEL
        self._base_url = base_url or settings.GEMINI_BASE_URL
        self._enabled = settings.GEMINI_ENABLED
        self._timeout = (
            getattr(settings, "GEMINI_TIMEOUT", 0.0)
            or settings.LLM_TIMEOUT
        )
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
                "Gemini provider is disabled.",
                provider=self.name,
            )
        if not self._api_key:
            raise ProviderConfigError(
                "Gemini API key is not configured.",
                provider=self.name,
            )

        url = (
            f"{self._base_url}/models/{self._model}"
            f":generateContent?key={self._api_key}"
        )
        payload = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }

        timeout = kwargs.get("timeout", self._timeout)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(
                f"Gemini request timed out: {exc}",
                provider=self.name,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"Gemini network error: {exc}",
                provider=self.name,
            ) from exc

        if response.status_code == 400 or response.status_code in (401, 403):
            raise ProviderAuthError(
                "Gemini authentication failed (invalid API key).",
                provider=self.name,
            )
        if response.status_code >= 500:
            raise ProviderUnavailableError(
                f"Gemini returned HTTP {response.status_code}",
                provider=self.name,
            )
        if response.status_code == 429:
            raise ProviderUnavailableError(
                "Gemini rate limit exceeded.",
                provider=self.name,
            )
        if response.status_code != 200:
            raise ProviderUnavailableError(
                f"Gemini returned HTTP {response.status_code}",
                provider=self.name,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderUnavailableError(
                "Gemini returned a malformed response",
                provider=self.name,
            ) from exc

        text = ""

        candidates = data.get("candidates", [])
        if candidates:
            parts = (
                candidates[0]
                .get("content", {})
                .get("parts", [])
            )
            if parts:
                text = parts[0].get("text", "").strip()

        self.mark_success()

        return LLMResponse(
            text=text,
            provider=self.name,
            model=self._model,
            usage=data.get("usageMetadata", {}),
            raw=data,
        )


gemini_provider = GeminiProvider()
