"""Ollama LLM provider — local inference, zero quota risk."""

import httpx

from app.core.config import settings
from app.services.llm.base import (
    LLMProvider,
    LLMResponse,
    ProviderUnavailableError,
)


class OllamaProvider(LLMProvider):
    """Ollama local inference provider."""

    name = "ollama"

    def __init__(self, base_url: str | None = None, model: str | None = None):
        # Allow tests to inject config without touching the global
        # settings singleton.
        self._base_url = base_url or settings.OLLAMA_BASE_URL
        self._model = model or settings.OLLAMA_MODEL
        self._enabled = settings.OLLAMA_ENABLED
        self._timeout = (
            getattr(settings, "OLLAMA_TIMEOUT", 0.0)
            or settings.LLM_TIMEOUT
        )
        self._available: bool | None = None

    def is_available(self) -> bool:
        if not self._enabled:
            return False
        if self._available is not None:
            return self._available
        try:
            r = httpx.get(
                f"{self._base_url}/api/tags",
                timeout=3.0,
            )
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def reset_availability(self):
        self._available = None

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        **kwargs,
    ) -> LLMResponse:
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                # Pass max_tokens through as num_predict so long
                # structured output (JSON) is not truncated.
                "num_predict": max_tokens,
            },
        }

        timeout = kwargs.get("timeout", self._timeout)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(
                f"Ollama request timed out: {exc}",
                provider=self.name,
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderUnavailableError(
                f"Ollama returned HTTP {exc.response.status_code}",
                provider=self.name,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"Ollama network error: {exc}",
                provider=self.name,
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderUnavailableError(
                "Ollama returned a malformed response",
                provider=self.name,
            ) from exc

        text = data.get("response", "").strip()

        return LLMResponse(
            text=text,
            provider=self.name,
            model=self._model,
            raw=data,
        )


# ---------------------------------------------------------------------------
# Singleton used by the provider registry
# ---------------------------------------------------------------------------
ollama_provider = OllamaProvider()


# ---------------------------------------------------------------------------
# Legacy helper — kept for backward compatibility with existing callers
# that import ``from app.services.llm.ollama import generate_answer``.
# ---------------------------------------------------------------------------

async def generate_answer(prompt: str) -> str:
    """Generate an answer using the Ollama provider.

    This is a convenience wrapper.  New code should prefer
    ``from app.services.llm import get_llm_answer`` instead.
    """
    resp = await ollama_provider.generate(prompt)
    if not resp.text:
        return (
            "The available research context is insufficient "
            "to identify a research gap."
        )
    return resp.text
