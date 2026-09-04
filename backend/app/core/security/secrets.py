"""Secrets management — never log or expose API keys.

Per the architecture (Section 24), all API keys live in platform
environment variables or .env — never in source control. This
module provides safe helpers for accessing secrets.
"""

import logging


logger = logging.getLogger(__name__)


def mask_secret(value: str, visible_chars: int = 4) -> str:
    """Mask a secret value for safe logging.

    Example: "sk-abc123def456" → "sk-****def456"
    """
    if not value:
        return "(not set)"
    if len(value) <= visible_chars:
        return "****"
    return (
        value[:visible_chars]
        + "****"
        + value[-visible_chars:]
    )


def validate_required_secrets() -> list[str]:
    """Check that critical secrets are configured.

    Returns a list of warnings for missing secrets.
    Does NOT return the actual values.
    """
    from app.core.config import settings

    warnings = []

    if not settings.active_database_url:
        warnings.append("DATABASE_URL is not configured")

    llm_chain = settings.llm_fallback_chain
    has_any_llm = False
    for provider in llm_chain:
        if provider == "ollama":
            has_any_llm = True  # Ollama needs no key
        elif provider == "groq" and settings.GROQ_API_KEY:
            has_any_llm = True
        elif provider == "gemini" and settings.GEMINI_API_KEY:
            has_any_llm = True
        elif provider == "openrouter" and settings.OPENROUTER_API_KEY:
            has_any_llm = True

    if not has_any_llm:
        warnings.append(
            "No LLM provider API keys configured. "
            "Only Ollama (local) will be available."
        )

    return warnings
