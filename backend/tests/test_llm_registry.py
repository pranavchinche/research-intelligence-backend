"""Tests for the LLM provider registry and fallback chain.

These use lightweight in-memory fake providers so no network or
real API calls are made.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.services.llm.base import (
    LLMProvider,
    LLMResponse,
    ProviderError,
    ProviderUnavailableError,
)
from app.services.llm.provider_registry import ProviderRegistry


class FakeProvider(LLMProvider):
    """Configurable fake provider for registry tests."""

    def __init__(
        self,
        name: str,
        available: bool = True,
        text: str = "ok",
        failures_before_success: int = 0,
    ):
        self.name = name
        self._available = available
        self._text = text
        self._failures_before_success = failures_before_success
        self._calls = 0
        self._enabled = True

    def is_available(self) -> bool:
        return self._enabled and self._available

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        **kwargs,
    ) -> LLMResponse:
        self._calls += 1
        if self._failures_before_success > 0:
            self._failures_before_success -= 1
            raise ProviderUnavailableError("boom", provider=self.name)
        return LLMResponse(
            text=self._text,
            provider=self.name,
        )


class FailingProvider(LLMProvider):
    """Always raises a provider error (used for fallback tests)."""

    def __init__(self, name: str):
        self.name = name
        self._enabled = True
        self._calls = 0

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt, max_tokens=4096, temperature=0.1, **kwargs):
        self._calls += 1
        raise ProviderUnavailableError("down", provider=self.name)


class EmptyProvider(LLMProvider):
    """Returns an empty text response (permanent outcome)."""

    def __init__(self, name: str):
        self.name = name
        self._enabled = True
        self._calls = 0

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt, max_tokens=4096, temperature=0.1, **kwargs):
        self._calls += 1
        return LLMResponse(text="", provider=self.name)


# ---------------------------------------------------------------------------
# 1. Provider registration
# ---------------------------------------------------------------------------

def test_register_and_get():
    reg = ProviderRegistry(providers={}, chain=["foo"])
    fake = FakeProvider("foo")
    reg.register("foo", fake)
    assert reg.get("foo") is fake
    assert reg.get("missing") is None


def test_register_in_chain_appends():
    reg = ProviderRegistry(providers={}, chain=["a"])
    reg.register("a", FakeProvider("a"))
    reg.register("b", FakeProvider("b"), in_chain=True)
    assert reg.chain == ["a", "b"]


# ---------------------------------------------------------------------------
# 2. Provider selection logic
# ---------------------------------------------------------------------------

def test_select_default_provider():
    reg = ProviderRegistry(
        providers={"groq": FakeProvider("groq")},
        chain=["groq"],
    )
    # default_provider comes from settings.LLM_DEFAULT_PROVIDER (ollama),
    # which is not registered here -> select_provider returns None.
    assert reg.select_provider() is None


def test_select_default_when_registered():
    fake = FakeProvider("ollama")
    reg = ProviderRegistry(providers={"ollama": fake}, chain=["ollama"])
    assert reg.select_provider() == "ollama"


def test_get_active_provider_name():
    reg = ProviderRegistry(
        providers={"a": FakeProvider("a", available=False)},
        chain=["a", "b"],
    )
    reg.register("b", FakeProvider("b", available=True), in_chain=True)
    assert reg.get_active_provider_name() == "b"


def test_enabled_providers_list():
    reg = ProviderRegistry(
        providers={
            "a": FakeProvider("a", available=True),
            "b": FakeProvider("b", available=False),
        },
        chain=["a", "b"],
    )
    assert set(reg.enabled_providers) == {"a"}


# ---------------------------------------------------------------------------
# 3. Disabled provider
# ---------------------------------------------------------------------------

async def test_disabled_provider_skipped():
    disabled = FakeProvider("a", available=True)
    disabled._enabled = False
    ok = FakeProvider("b")
    reg = ProviderRegistry(
        providers={"a": disabled, "b": ok},
        chain=["a", "b"],
    )
    response = await reg.generate("hi")
    assert response.provider == "b"
    assert disabled._calls == 0


# ---------------------------------------------------------------------------
# 4. Successful provider response
# ---------------------------------------------------------------------------

async def test_successful_response():
    ok = FakeProvider("a")
    reg = ProviderRegistry(providers={"a": ok}, chain=["a"])
    response = await reg.generate("hi")
    assert response.provider == "a"
    assert response.text == "ok"
    assert reg.last_used_provider == "a"


# ---------------------------------------------------------------------------
# 5. Provider timeout / failure -> fallback to next provider
# ---------------------------------------------------------------------------

async def test_failure_falls_back_to_next():
    fail = FailingProvider("a")
    ok = FakeProvider("b")
    reg = ProviderRegistry(
        providers={"a": fail, "b": ok},
        chain=["a", "b"],
    )
    response = await reg.generate("hi")
    assert response.provider == "b"
    assert fail._calls == 1
    assert reg.last_used_provider == "b"


async def test_empty_message_falls_back():
    empty = EmptyProvider("a")
    ok = FakeProvider("b")
    reg = ProviderRegistry(
        providers={"a": empty, "b": ok},
        chain=["a", "b"],
    )
    response = await reg.generate("hi")
    assert response.provider == "b"


# ---------------------------------------------------------------------------
# 6. Ollama fallback (ollama is last in chain, reached only after others fail)
# ---------------------------------------------------------------------------

async def test_ollama_fallback():
    fail = FailingProvider("groq")
    fail2 = FailingProvider("gemini")
    ollama_ok = FakeProvider("ollama")
    reg = ProviderRegistry(
        providers={"groq": fail, "gemini": fail2, "ollama": ollama_ok},
        chain=["groq", "gemini", "ollama"],
    )
    response = await reg.generate("hi")
    assert response.provider == "ollama"
    assert reg.last_used_provider == "ollama"


# ---------------------------------------------------------------------------
# 7. All providers unavailable / failed -> ProviderError
# ---------------------------------------------------------------------------

async def test_all_providers_unavailable():
    reg = ProviderRegistry(
        providers={
            "a": FakeProvider("a", available=False),
            "b": FakeProvider("b", available=False),
        },
        chain=["a", "b"],
    )
    with pytest.raises(ProviderError):
        await reg.generate("hi")


async def test_all_providers_failed():
    reg = ProviderRegistry(
        providers={"a": FailingProvider("a"), "b": FailingProvider("b")},
        chain=["a", "b"],
    )
    with pytest.raises(ProviderError):
        await reg.generate("hi")


# ---------------------------------------------------------------------------
# No retry loop: each provider attempted exactly once
# ---------------------------------------------------------------------------

async def test_no_retry_loop():
    fail = FailingProvider("a")
    reg = ProviderRegistry(providers={"a": fail}, chain=["a"])
    with pytest.raises(ProviderError):
        await reg.generate("hi")
    assert fail._calls == 1


async def test_unknown_provider_in_chain_skipped():
    ok = FakeProvider("b")
    reg = ProviderRegistry(
        providers={"b": ok},
        chain=["missing", "b"],
    )
    response = await reg.generate("hi")
    assert response.provider == "b"
