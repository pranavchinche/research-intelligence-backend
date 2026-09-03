"""Pluggable LLM provider interface and registry.

Per the architecture (Section 17), every LLM call goes through
``provider_registry.generate(prompt)`` — callers never import a
specific provider directly.  The registry tries providers in
priority order, skipping any that report ``is_available() == False``.
"""

from app.services.llm.base import (
    LLMProvider,
    LLMResponse,
    ProviderError,
    ProviderUnavailableError,
    ProviderAuthError,
    ProviderConfigError,
)
from app.services.llm.provider_registry import (
    provider_registry,
    get_llm_answer,
)

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "ProviderError",
    "ProviderUnavailableError",
    "ProviderAuthError",
    "ProviderConfigError",
    "provider_registry",
    "get_llm_answer",
]
