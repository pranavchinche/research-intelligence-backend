"""Tests for the individual LLM providers.

Providers are exercised with mocked HTTP responses (respx) and
injected configuration so no real/paid API calls are made.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import respx
import httpx

from app.services.llm import (
    ProviderConfigError,
    ProviderAuthError,
    ProviderUnavailableError,
)
from app.services.llm.groq import GroqProvider
from app.services.llm.gemini import GeminiProvider
from app.services.llm.openrouter import OpenRouterProvider
from app.services.llm.ollama import OllamaProvider


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

@respx.mock
async def test_ollama_success():
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(
            200,
            json={"response": " ollama answer "},
        )
    )
    provider = OllamaProvider()
    response = await provider.generate("hello")
    assert response.text == "ollama answer"
    assert response.provider == "ollama"


@respx.mock
async def test_ollama_unavailable():
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(500)
    )
    provider = OllamaProvider()
    with pytest.raises(ProviderUnavailableError):
        await provider.generate("hello")


@respx.mock
async def test_ollama_timeout():
    def _timeout(request):
        raise httpx.ReadTimeout("timed out")

    respx.post(
        "http://localhost:11434/api/generate"
    ).mock(side_effect=_timeout)
    provider = OllamaProvider()
    with pytest.raises(ProviderUnavailableError):
        await provider.generate("hello")


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------

async def test_groq_missing_api_key():
    provider = GroqProvider(api_key="")
    assert provider.is_available() is False
    with pytest.raises(ProviderConfigError):
        await provider.generate("hello")


async def test_gemini_missing_api_key():
    provider = GeminiProvider(api_key="")
    assert provider.is_available() is False
    with pytest.raises(ProviderConfigError):
        await provider.generate("hello")


async def test_openrouter_missing_api_key():
    provider = OpenRouterProvider(api_key="")
    assert provider.is_available() is False
    with pytest.raises(ProviderConfigError):
        await provider.generate("hello")


# ---------------------------------------------------------------------------
# Disabled provider
# ---------------------------------------------------------------------------

async def test_groq_disabled():
    provider = GroqProvider(api_key="some-key")
    provider._enabled = False
    assert provider.is_available() is False
    with pytest.raises(ProviderConfigError):
        await provider.generate("hello")


# ---------------------------------------------------------------------------
# Successful responses
# ---------------------------------------------------------------------------

@respx.mock
async def test_groq_success():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": " groq answer "}}
                ],
                "usage": {"total_tokens": 10},
            },
        )
    )
    provider = GroqProvider(api_key="test-key")
    response = await provider.generate("hello")
    assert response.text == "groq answer"
    assert response.provider == "groq"
    assert response.usage == {"total_tokens": 10}


@respx.mock
async def test_gemini_success():
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-1.5-flash:generateContent?key=test-key"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": " gemini answer "}]
                        }
                    }
                ],
                "usageMetadata": {"totalTokenCount": 5},
            },
        )
    )
    provider = GeminiProvider(api_key="test-key")
    response = await provider.generate("hello")
    assert response.text == "gemini answer"
    assert response.provider == "gemini"


@respx.mock
async def test_openrouter_success():
    respx.post(
        "https://openrouter.ai/api/v1/chat/completions"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": " openrouter answer "}}
                ]
            },
        )
    )
    provider = OpenRouterProvider(api_key="test-key")
    response = await provider.generate("hello")
    assert response.text == "openrouter answer"
    assert response.provider == "openrouter"


# ---------------------------------------------------------------------------
# Transient failures -> ProviderUnavailableError
# ---------------------------------------------------------------------------

@respx.mock
async def test_groq_server_error():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(500)
    )
    provider = GroqProvider(api_key="test-key")
    with pytest.raises(ProviderUnavailableError):
        await provider.generate("hello")


@respx.mock
async def test_groq_timeout():
    def _timeout(request):
        raise httpx.ReadTimeout("timed out")

    respx.post(
        "https://api.groq.com/openai/v1/chat/completions"
    ).mock(side_effect=_timeout)
    provider = GroqProvider(api_key="test-key")
    with pytest.raises(ProviderUnavailableError):
        await provider.generate("hello")


@respx.mock
async def test_groq_rate_limit():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(429)
    )
    provider = GroqProvider(api_key="test-key")
    with pytest.raises(ProviderUnavailableError):
        await provider.generate("hello")


@respx.mock
async def test_groq_network_error():
    respx.post(
        "https://api.groq.com/openai/v1/chat/completions"
    ).mock(side_effect=httpx.ConnectError("no route"))
    provider = GroqProvider(api_key="test-key")
    with pytest.raises(ProviderUnavailableError):
        await provider.generate("hello")


# ---------------------------------------------------------------------------
# Auth failure -> ProviderAuthError
# ---------------------------------------------------------------------------

@respx.mock
async def test_groq_invalid_api_key():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(401)
    )
    provider = GroqProvider(api_key="bad-key")
    with pytest.raises(ProviderAuthError):
        await provider.generate("hello")


@respx.mock
async def test_gemini_invalid_api_key():
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-1.5-flash:generateContent?key=bad-key"
    ).mock(return_value=httpx.Response(403))
    provider = GeminiProvider(api_key="bad-key")
    with pytest.raises(ProviderAuthError):
        await provider.generate("hello")


# ---------------------------------------------------------------------------
# Malformed response -> ProviderUnavailableError
# ---------------------------------------------------------------------------

@respx.mock
async def test_groq_malformed_response():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, text="not-json{{{")
    )
    provider = GroqProvider(api_key="test-key")
    with pytest.raises(ProviderUnavailableError):
        await provider.generate("hello")
