#backend/app/services/embedding_service.py
import asyncio
import threading
from functools import partial

from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(MODEL_NAME)
    return _model


def _encode_sync(text: str) -> list[float]:

    if not text or not text.strip():
        raise ValueError(
            "Cannot generate embedding for empty text."
        )

    text = text.replace("\x00", "")

    model = _get_model()

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
    _get_model()
    return await loop.run_in_executor(
        None, partial(_encode_sync, text)
    )
