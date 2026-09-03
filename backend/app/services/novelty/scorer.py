import hashlib

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.paper import Paper
from app.models.paper_chunk import PaperChunk


def cosine_similarity(
    embedding_a,
    embedding_b,
) -> float:
    """
    Calculate cosine similarity between two embedding vectors.
    """

    if not embedding_a or not embedding_b:
        return 0.0

    if len(embedding_a) != len(embedding_b):
        return 0.0

    dot_product = sum(
        a * b
        for a, b in zip(
            embedding_a,
            embedding_b,
        )
    )

    norm_a = sum(
        a * a
        for a in embedding_a
    ) ** 0.5

    norm_b = sum(
        b * b
        for b in embedding_b
    ) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (
        norm_a * norm_b
    )


def _hash_full_text(
    full_text: str | None,
) -> str | None:
    """
    Compute a SHA256 hex digest of a paper's full_text.

    Returns None when full_text is NULL/empty. A missing full_text
    can never be safely treated as an exact match for anything -
    we must not accidentally group two papers together just because
    both happen to have no stored text.
    """

    if not full_text:
        return None

    return hashlib.sha256(
        full_text.encode("utf-8")
    ).hexdigest()


async def _find_exact_duplicate_paper_ids(
    db: AsyncSession,
    paper_id: int,
    target_full_text: str | None,
):
    """
    Identify OTHER papers whose full_text is an EXACT SHA256 match
    for the target paper's full_text.

    WHY exact duplicates must be excluded from novelty comparisons:

    Novelty is meant to measure how similar the target paper is to
    the *independent* body of existing work. A paper that is a
    verbatim duplicate of the target (identical full_text,
    byte-for-byte) is not an independent data point - it is
    effectively the same document appearing twice in the corpus.
    Comparing the target against its own duplicate would trivially
    produce ~1.0 similarity and collapse the novelty score to ~0.0,
    which misrepresents how novel the paper actually is relative to
    genuinely distinct prior work.

    This check is intentionally narrow and mechanical:
    - It only compares SHA256(full_text) values.
    - It does NOT look at titles, metadata, or "roughly similar"
      text - only an identical hash counts. Merely similar titles
      must never trigger this exclusion.
    - Papers with a NULL/empty full_text are never considered a
      duplicate of anything (see _hash_full_text()), so this can
      never misfire on missing data.

    Returns:
        (duplicate_paper_ids, excluded_duplicates)

        duplicate_paper_ids: a set[int] of other paper ids to
            exclude from novelty comparisons.
        excluded_duplicates: a list[dict] describing the excluded
            papers, suitable for inclusion in the API response.
    """

    target_hash = _hash_full_text(
        target_full_text
    )

    # If the target paper itself has no full_text, there is nothing
    # reliable to hash-compare against - skip duplicate detection
    # entirely rather than guessing.
    if target_hash is None:
        return set(), []

    # Only load the columns actually needed (id + full_text) for
    # other papers, instead of full ORM rows or embeddings, since
    # this is just a hash check - avoids unnecessary data loading.
    other_papers_stmt = (
        select(
            Paper.id,
            Paper.full_text,
        )
        .where(
            Paper.id != paper_id,
        )
    )

    result = await db.execute(other_papers_stmt)

    other_papers = result.all()

    duplicate_paper_ids = set()

    excluded_duplicates = []

    for other_paper_id, other_full_text in other_papers:

        other_hash = _hash_full_text(
            other_full_text
        )

        # Exact duplicate means: SHA256(full_text) is identical.
        # A None hash (missing full_text) never counts as a match,
        # even against another None hash.
        if other_hash is not None and other_hash == target_hash:

            duplicate_paper_ids.add(
                other_paper_id
            )

            excluded_duplicates.append(
                {
                    "paper_id": other_paper_id,
                    "reason": "Exact full-text duplicate",
                }
            )

    return duplicate_paper_ids, excluded_duplicates


async def calculate_novelty(
    db: AsyncSession,
    paper_id: int,
):
    # --------------------------------------------------
    # GET TARGET PAPER CHUNKS
    # --------------------------------------------------

    target_stmt = (
        select(PaperChunk)
        .where(
            PaperChunk.paper_id == paper_id,
            PaperChunk.embedding.is_not(None),
        )
    )

    result = await db.execute(target_stmt)

    target_chunks = result.scalars().all()

    if not target_chunks:
        return {
            "paper_id": paper_id,
            "novelty_score": None,
            "highest_similarity": None,
            "message": (
                "No embeddings available for this paper."
            ),
            "comparisons": [],
            "excluded_duplicates": [],
        }

    # --------------------------------------------------
    # IDENTIFY EXACT DUPLICATE PAPERS (BY FULL_TEXT SHA256)
    #
    # This runs before any other paper's chunks/embeddings are
    # loaded, so duplicate papers can be filtered directly out of
    # the chunk query below instead of being fetched and discarded
    # afterward.
    # --------------------------------------------------

    target_paper_stmt = (
        select(Paper.full_text)
        .where(Paper.id == paper_id)
    )

    result = await db.execute(target_paper_stmt)

    target_full_text_row = result.first()

    target_full_text = (
        target_full_text_row[0]
        if target_full_text_row is not None
        else None
    )

    duplicate_paper_ids, excluded_duplicates = (
        await _find_exact_duplicate_paper_ids(
            db,
            paper_id,
            target_full_text,
        )
    )

    # --------------------------------------------------
    # GET CHUNKS FROM OTHER PAPERS
    #
    # The target paper is never compared with itself (the
    # paper_id != paper_id filter, unchanged), and any paper that
    # is an exact full-text duplicate of the target is excluded
    # here too, so it can never contribute a comparison or affect
    # the novelty score.
    # --------------------------------------------------

    other_stmt = (
        select(PaperChunk)
        .where(
            PaperChunk.paper_id != paper_id,
            PaperChunk.embedding.is_not(None),
        )
    )

    if duplicate_paper_ids:
        other_stmt = other_stmt.where(
            PaperChunk.paper_id.not_in(
                duplicate_paper_ids
            )
        )

    result = await db.execute(other_stmt)

    other_chunks = result.scalars().all()

    if not other_chunks:

        if excluded_duplicates:
            # There WERE other papers, but every one of them was an
            # exact full-text duplicate of the target paper, so none
            # of them counts as independent prior work.
            message = (
                "Novelty cannot be estimated because the only "
                "other paper(s) found are exact full-text "
                "duplicates of this paper, not independent "
                "research."
            )
        else:
            message = (
                "Novelty cannot be estimated because "
                "no other papers are available for comparison."
            )

        return {
            "paper_id": paper_id,
            "novelty_score": None,
            "highest_similarity": None,
            "message": message,
            "comparisons": [],
            "excluded_duplicates": excluded_duplicates,
        }

    # --------------------------------------------------
    # GROUP OTHER CHUNKS BY PAPER
    # --------------------------------------------------

    papers = {}

    for chunk in other_chunks:
        papers.setdefault(
            chunk.paper_id,
            []
        ).append(chunk)

    comparisons = []

    # --------------------------------------------------
    # COMPARE TARGET PAPER WITH EACH OTHER PAPER
    # --------------------------------------------------

    for other_paper_id, chunks in papers.items():

        best_similarities = []

        for target_chunk in target_chunks:

            best_similarity = 0.0

            for other_chunk in chunks:

                similarity = cosine_similarity(
                    target_chunk.embedding,
                    other_chunk.embedding,
                )

                if similarity > best_similarity:
                    best_similarity = similarity

            best_similarities.append(
                best_similarity
            )

        # --------------------------------------------------
        # AVERAGE STRONGEST CHUNK MATCHES
        # --------------------------------------------------

        if best_similarities:

            paper_similarity = (
                sum(best_similarities)
                / len(best_similarities)
            )

        else:
            paper_similarity = 0.0

        comparisons.append(
            {
                "paper_id": other_paper_id,
                "similarity": round(
                    paper_similarity,
                    4,
                ),
            }
        )

    # --------------------------------------------------
    # SORT BY SIMILARITY
    # --------------------------------------------------

    comparisons.sort(
        key=lambda x: x["similarity"],
        reverse=True,
    )

    if not comparisons:
        return {
            "paper_id": paper_id,
            "novelty_score": None,
            "highest_similarity": None,
            "message": (
                "No valid paper comparisons available."
            ),
            "comparisons": [],
            "excluded_duplicates": excluded_duplicates,
        }

    # --------------------------------------------------
    # HIGHEST SIMILARITY
    # --------------------------------------------------

    highest_similarity = comparisons[0]["similarity"]

    # --------------------------------------------------
    # NOVELTY SCORE
    #
    # High similarity  -> low novelty
    # Low similarity   -> high novelty
    #
    # Range: 0.0 - 1.0
    #
    # NOTE: this is computed only from comparisons against
    # independent (non-duplicate) papers, so an exact duplicate can
    # never artificially deflate this score.
    # --------------------------------------------------

    novelty_score = (
        1.0 - highest_similarity
    )

    novelty_score = max(
        0.0,
        min(
            1.0,
            novelty_score,
        ),
    )

    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

    return {
        "paper_id": paper_id,
        "novelty_score": round(
            novelty_score,
            4,
        ),
        "highest_similarity": round(
            highest_similarity,
            4,
        ),
        "comparisons": comparisons[:5],
        "excluded_duplicates": excluded_duplicates,
    }