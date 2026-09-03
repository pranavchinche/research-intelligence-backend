#backend/app/services/embedding_service.py
import asyncio
from functools import partial

from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def _encode_sync(text: str) -> list[float]:

    if not text or not text.strip():
        raise ValueError(
            "Cannot generate embedding for empty text."
        )

    text = text.replace("\x00", "")

    embedding = model.encode(
        text,
        normalize_embeddings=True,
    )

    if embedding is None:
        raise ValueError(
            "Embedding model returned None."
        )

    result = embedding.tolist()

    if not result:
        raise ValueError(
            "Embedding model returned an empty vector."
        )

    return result


def generate_embedding(text: str) -> list[float]:
    """Synchronous embedding generation for use in sync contexts."""
    return _encode_sync(text)


async def generate_embedding_async(text: str) -> list[float]:
    """Non-blocking embedding generation for async contexts.

    Runs the CPU-bound sentence-transformers encode() in a
    thread pool so the asyncio event loop is never blocked.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, partial(_encode_sync, text)
    )