"""Cache key generation helpers.

Per the architecture (Section 23), cache keys use normalised
query+source+page for search results, paper_id for LLM results,
content hash for embeddings.
"""

import hashlib
import json


def _hash(data: str) -> str:
    """SHA-256 hash for compact, collision-resistant keys."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def search_cache_key(
    query: str,
    source: str,
    page: int = 1,
) -> str:
    """Cache key for arXiv/OpenAlex search results."""
    normalised = " ".join(query.lower().split())
    return f"search:{source}:{_hash(normalised)}:p{page}"


def paper_metadata_key(external_id: str) -> str:
    """Cache key for single paper metadata lookup."""
    return f"paper:meta:{_hash(external_id)}"


def embedding_cache_key(text_hash: str) -> str:
    """Cache key for embedding by content hash."""
    return f"embed:{text_hash}"


def llm_chat_cache_key(
    paper_ids: list[int],
    question: str,
    model: str,
    provider: str,
) -> str:
    """Cache key for LLM chat responses (short TTL)."""
    normalised_q = " ".join(question.lower().split())
    ids_str = ",".join(str(i) for i in sorted(paper_ids))
    combined = f"{ids_str}:{normalised_q}:{model}:{provider}"
    return f"llm:chat:{_hash(combined)}"


def llm_analysis_cache_key(
    paper_id: int,
    analysis_type: str,
    provider: str,
) -> str:
    """Cache key for LLM analysis (summary/gaps/strengths)."""
    return f"llm:{analysis_type}:{paper_id}:{provider}"


def related_papers_cache_key(paper_id: int) -> str:
    """Cache key for related-paper recommendations."""
    return f"related:{paper_id}"


def health_cache_key(service: str) -> str:
    """Cache key for provider/service health status."""
    return f"health:{service}"
