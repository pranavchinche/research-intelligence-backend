import hashlib
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.paper import Paper
from app.models.paper_chunk import PaperChunk


def _sha256_text(text: str | None) -> str | None:
    if not text:
        return None

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _cosine_similarity(
    embedding_a: list[float],
    embedding_b: list[float],
) -> float:
    """
    Calculate cosine similarity for already-normalized embeddings.

    The embedding service uses normalize_embeddings=True, so the
    dot product is equivalent to cosine similarity.
    """
    if not embedding_a or not embedding_b:
        return 0.0

    if len(embedding_a) != len(embedding_b):
        raise ValueError(
            "Embedding dimensions do not match."
        )

    similarity = sum(
        a * b
        for a, b in zip(embedding_a, embedding_b)
    )

    # Protect against tiny floating-point overflow.
    return max(
        0.0,
        min(1.0, float(similarity)),
    )


async def _get_paper(
    db: AsyncSession,
    paper_id: int,
) -> Paper | None:

    result = await db.execute(
        select(Paper).where(
            Paper.id == paper_id
        )
    )

    return result.scalar_one_or_none()


async def _get_embedded_chunks(
    db: AsyncSession,
    paper_id: int,
) -> list[PaperChunk]:

    result = await db.execute(
        select(PaperChunk)
        .where(
            PaperChunk.paper_id == paper_id,
            PaperChunk.embedding.is_not(None),
        )
        .order_by(PaperChunk.id)
    )

    return list(result.scalars().all())


def _embedding_to_list(
    embedding: Any,
) -> list[float]:

    if embedding is None:
        return []

    if isinstance(embedding, list):
        return [
            float(value)
            for value in embedding
        ]

    # pgvector / SQLAlchemy may return an object
    # that can be converted to a list.
    try:
        return [
            float(value)
            for value in embedding
        ]
    except TypeError:
        raise ValueError(
            "Unable to convert stored embedding to a list."
        )


def _best_matches(
    source_chunks: list[PaperChunk],
    target_chunks: list[PaperChunk],
) -> tuple[float, list[dict]]:

    if not source_chunks or not target_chunks:
        return 0.0, []

    best_similarities: list[float] = []
    evidence: list[dict] = []

    for source_chunk in source_chunks:

        source_embedding = _embedding_to_list(
            source_chunk.embedding
        )

        best_similarity = -1.0
        best_target = None

        for target_chunk in target_chunks:

            target_embedding = _embedding_to_list(
                target_chunk.embedding
            )

            similarity = _cosine_similarity(
                source_embedding,
                target_embedding,
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_target = target_chunk

        if best_target is None:
            continue

        best_similarities.append(
            best_similarity
        )

        evidence.append(
            {
                "paper_1_chunk_id": source_chunk.chunk_id,
                "paper_1_page": source_chunk.page_number,
                "paper_2_chunk_id": best_target.chunk_id,
                "paper_2_page": best_target.page_number,
                "similarity": round(
                    best_similarity,
                    4,
                ),
            }
        )

    if not best_similarities:
        return 0.0, []

    average_similarity = (
        sum(best_similarities)
        / len(best_similarities)
    )

    return average_similarity, evidence


async def compare_papers(
    db: AsyncSession,
    paper_id_1: int,
    paper_id_2: int,
) -> dict:

    if paper_id_1 == paper_id_2:
        raise ValueError(
            "Cannot compare a paper with itself."
        )

    paper_1 = await _get_paper(
        db,
        paper_id_1,
    )

    if paper_1 is None:
        raise ValueError(
            f"Paper {paper_id_1} not found."
        )

    paper_2 = await _get_paper(
        db,
        paper_id_2,
    )

    if paper_2 is None:
        raise ValueError(
            f"Paper {paper_id_2} not found."
        )

    # ---------------------------------------------------------
    # EXACT DUPLICATE DETECTION
    # ---------------------------------------------------------

    hash_1 = _sha256_text(
        paper_1.full_text
    )

    hash_2 = _sha256_text(
        paper_2.full_text
    )

    is_exact_duplicate = (
        hash_1 is not None
        and hash_2 is not None
        and hash_1 == hash_2
    )

    if is_exact_duplicate:

        return {
            "paper_1": {
                "paper_id": paper_1.id,
                "title": paper_1.title,
            },
            "paper_2": {
                "paper_id": paper_2.id,
                "title": paper_2.title,
            },
            "similarity": 1.0,
            "difference_score": 0.0,
            "is_exact_duplicate": True,
            "evidence": [],
        }

    # ---------------------------------------------------------
    # EMBEDDED CHUNKS
    # ---------------------------------------------------------

    chunks_1 = await _get_embedded_chunks(
        db,
        paper_id_1,
    )

    chunks_2 = await _get_embedded_chunks(
        db,
        paper_id_2,
    )

    if not chunks_1:
        raise ValueError(
            f"Paper {paper_id_1} has no embedded chunks."
        )

    if not chunks_2:
        raise ValueError(
            f"Paper {paper_id_2} has no embedded chunks."
        )

    # ---------------------------------------------------------
    # SYMMETRIC COMPARISON
    # ---------------------------------------------------------

    similarity_1_to_2, evidence_1 = _best_matches(
        chunks_1,
        chunks_2,
    )

    similarity_2_to_1, evidence_2 = _best_matches(
        chunks_2,
        chunks_1,
    )

    final_similarity = (
        similarity_1_to_2
        + similarity_2_to_1
    ) / 2.0

    final_similarity = max(
        0.0,
        min(1.0, final_similarity),
    )

    difference_score = (
        1.0 - final_similarity
    )

    # ---------------------------------------------------------
    # BUILD STRONGEST EVIDENCE
    # ---------------------------------------------------------

    combined_evidence = (
        evidence_1
        + [
            {
                "paper_1_chunk_id": item[
                    "paper_2_chunk_id"
                ],
                "paper_1_page": item[
                    "paper_2_page"
                ],
                "paper_2_chunk_id": item[
                    "paper_1_chunk_id"
                ],
                "paper_2_page": item[
                    "paper_1_page"
                ],
                "similarity": item[
                    "similarity"
                ],
            }
            for item in evidence_2
        ]
    )

    combined_evidence.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    # Avoid returning the same chunk pair repeatedly.
    unique_evidence = []

    seen_pairs = set()

    for item in combined_evidence:

        pair = (
            item["paper_1_chunk_id"],
            item["paper_2_chunk_id"],
        )

        if pair in seen_pairs:
            continue

        seen_pairs.add(pair)
        unique_evidence.append(item)

        if len(unique_evidence) >= 5:
            break

    return {
        "paper_1": {
            "paper_id": paper_1.id,
            "title": paper_1.title,
        },
        "paper_2": {
            "paper_id": paper_2.id,
            "title": paper_2.title,
        },
        "similarity": round(
            final_similarity,
            4,
        ),
        "difference_score": round(
            difference_score,
            4,
        ),
        "is_exact_duplicate": False,
        "evidence": unique_evidence,
    }