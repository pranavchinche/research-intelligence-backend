from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.paper_chunk import PaperChunk
from app.services.embedding_service import generate_embedding_async


# ============================================================
# RESEARCH TERMS
# ============================================================

RESEARCH_KEYWORDS = {
    "problem",
    "problems",
    "challenge",
    "challenges",
    "limitation",
    "limitations",
    "weakness",
    "weaknesses",
    "difficulty",
    "difficulties",
    "issue",
    "issues",
    "drawback",
    "drawbacks",
    "trade-off",
    "tradeoffs",

    "approach",
    "method",
    "methodology",
    "methodological",
    "technique",
    "framework",
    "system",
    "model",

    "architecture",
    "architectures",
    "network",
    "networks",
    "encoder",
    "decoder",
    "layer",
    "layers",
    "lstm",
    "rnn",
    "transformer",

    "result",
    "results",
    "performance",
    "evaluation",
    "experiment",
    "experiments",
    "accuracy",
    "bleu",
    "perplexity",
    "score",
    "scores",

    "dataset",
    "datasets",
    "training",
    "testing",
    "train",
    "test",

    "conclusion",
    "conclusions",
    "future",
    "work",
}


# ============================================================
# QUERY KEYWORD EXTRACTION
# ============================================================

def _tokenize(text: str) -> set[str]:
    """
    Lightweight tokenization.

    Keeps this dependency-free and avoids changing the
    existing project requirements.
    """

    cleaned = (
        text.lower()
        .replace("-", " ")
        .replace("_", " ")
    )

    words = set()

    for word in cleaned.split():

        word = word.strip(
            ".,?!:;()[]{}\"'"
        )

        if len(word) >= 3:
            words.add(word)

    return words


def keyword_score(
    text: str,
    query: str,
) -> float:
    """
    Calculate lexical overlap between the query and chunk.

    This is intentionally a secondary signal.
    Semantic similarity remains the primary signal.
    """

    text_words = _tokenize(text)
    query_words = _tokenize(query)

    if not query_words:
        return 0.0

    # --------------------------------------------------------
    # Direct query overlap
    # --------------------------------------------------------

    overlap = (
        text_words.intersection(query_words)
    )

    overlap_score = (
        len(overlap) / len(query_words)
    )

    # --------------------------------------------------------
    # Research terminology bonus
    # --------------------------------------------------------

    research_overlap = (
        text_words.intersection(
            RESEARCH_KEYWORDS
        )
        .intersection(query_words)
    )

    research_bonus = min(
        len(research_overlap) * 0.05,
        0.15,
    )

    return min(
        overlap_score * 0.35
        + research_bonus,
        0.50,
    )


# ============================================================
# LIMITATION DETECTION
# ============================================================

LIMITATION_PATTERNS = [
    "limitation",
    "limitations",
    "weakness",
    "weaknesses",
    "problem",
    "problems",
    "difficulty",
    "difficulties",
    "challenge",
    "challenges",
    "drawback",
    "drawbacks",
    "trade-off",
    "tradeoffs",
    "hurts",
    "degrades",
    "degradation",
    "poor performance",
    "worse performance",
    "cannot",
    "unable to",
    "fails",
    "failure",
    "not able to",
    "not possible",
    "difficult to",
    "remains difficult",
    "unresolved",
    "not clear",
    "lack of",
    "lack",
]


POSITIVE_ONLY_PATTERNS = [
    "useful property",
    "excellent performance",
    "powerful model",
    "powerful models",
    "achieved excellent",
    "successfully",
    "outperforms",
    "outperformed",
    "significantly outperform",
    "good performance",
]


def has_limitation_evidence(
    text: str,
) -> bool:
    """
    Detect whether a chunk contains limitation-related evidence.

    IMPORTANT:
    This function is NOT used by general RAG retrieval.

    It remains available for future gap/limitation-specific
    retrieval.
    """

    text_lower = text.lower()

    has_limitation = any(
        pattern in text_lower
        for pattern in LIMITATION_PATTERNS
    )

    if not has_limitation:
        return False

    positive_count = sum(
        pattern in text_lower
        for pattern in POSITIVE_ONLY_PATTERNS
    )

    limitation_count = sum(
        pattern in text_lower
        for pattern in LIMITATION_PATTERNS
    )

    if (
        positive_count > 0
        and limitation_count <= positive_count
    ):
        return False

    return True


# ============================================================
# RETRIEVAL
# ============================================================

async def retrieve_chunks(
    db: AsyncSession,
    query: str,
    paper_id: int | None = None,
    top_k: int = 10,
    min_similarity: float | None = None,
):

    top_k = min(
        max(top_k, 1),
        20,
    )

    query = query.strip()

    if not query:
        return []

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = await generate_embedding_async(
        query
    )

    distance = PaperChunk.embedding.cosine_distance(
        query_embedding
    )

    stmt = (
        select(
            PaperChunk,
            distance.label("distance"),
        )
        .where(
            PaperChunk.embedding.is_not(None)
        )
    )

    # --------------------------------------------------------
    # Optional paper restriction
    # --------------------------------------------------------

    if paper_id is not None:

        stmt = stmt.where(
            PaperChunk.paper_id == paper_id
        )

    # --------------------------------------------------------
    # Candidate pool
    #
    # Retrieve more candidates than we finally return.
    # This gives the hybrid ranking enough material.
    # --------------------------------------------------------

    candidate_limit = max(
        top_k * 5,
        30,
    )

    candidate_limit = min(
        candidate_limit,
        100,
    )

    stmt = (
        stmt
        .order_by(distance)
        .limit(candidate_limit)
    )

    result = await db.execute(
        stmt
    )

    rows = result.all()

    candidates = []

    # --------------------------------------------------------
    # Hybrid ranking
    # --------------------------------------------------------

    for chunk, distance_value in rows:

        distance_value = float(
            distance_value
        )

        semantic_similarity = (
            1.0 - distance_value
        )

        # Guard against tiny numerical errors.
        semantic_similarity = max(
            0.0,
            min(
                1.0,
                semantic_similarity,
            ),
        )

        kw_score = keyword_score(
            chunk.text,
            query,
        )

        # Semantic similarity remains dominant.
        final_score = (
            semantic_similarity * 0.80
            + kw_score * 0.20
        )

        candidates.append(
            {
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "page_number": chunk.page_number,
                "text": chunk.text,
                "model": chunk.model,
                "distance": distance_value,
                "similarity": semantic_similarity,
                "keyword_score": kw_score,
                "final_score": final_score,
                "is_neighbor": False,
            }
        )

    # --------------------------------------------------------
    # Sort by strongest evidence
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: item["final_score"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Similarity threshold
    # --------------------------------------------------------

    retrieved_chunks = []

    for chunk in candidates:

        if (
            min_similarity is not None
            and chunk["similarity"] < min_similarity
        ):
            continue

        print(
            f"Chunk {chunk['chunk_id']} | "
            f"Semantic: {chunk['similarity']:.4f} | "
            f"Keyword: {chunk['keyword_score']:.4f} | "
            f"Final: {chunk['final_score']:.4f}"
        )

        retrieved_chunks.append(
            chunk
        )

        if len(retrieved_chunks) >= top_k:
            break

    return retrieved_chunks


# ============================================================
# NEIGHBORING CHUNK EXPANSION
# ============================================================

async def expand_with_neighbors(
    db: AsyncSession,
    retrieved_chunks: list[dict],
    window: int = 1,
) -> list[dict]:

    if not retrieved_chunks:
        return []

    expanded = []

    seen = set()

    for chunk in retrieved_chunks:

        paper_id = chunk["paper_id"]
        chunk_id = chunk["chunk_id"]

        # ----------------------------------------------------
        # Original chunk
        # ----------------------------------------------------

        key = (
            paper_id,
            chunk_id,
        )

        if key not in seen:

            expanded.append(
                chunk
            )

            seen.add(key)

        # ----------------------------------------------------
        # Neighbor range
        # ----------------------------------------------------

        min_chunk_id = max(
            0,
            chunk_id - window,
        )

        max_chunk_id = (
            chunk_id + window
        )

        neighbor_stmt = (
            select(PaperChunk)
            .where(
                PaperChunk.paper_id == paper_id,
                PaperChunk.chunk_id >= min_chunk_id,
                PaperChunk.chunk_id <= max_chunk_id,
            )
            .order_by(
                PaperChunk.chunk_id
            )
        )

        result = await db.execute(
            neighbor_stmt
        )

        neighbors = result.scalars().all()

        for neighbor in neighbors:

            neighbor_key = (
                neighbor.paper_id,
                neighbor.chunk_id,
            )

            if neighbor_key in seen:
                continue

            expanded.append(
                {
                    "chunk_id": neighbor.chunk_id,
                    "paper_id": neighbor.paper_id,
                    "page_number": neighbor.page_number,
                    "text": neighbor.text,
                    "model": neighbor.model,
                    "distance": None,
                    "similarity": None,
                    "keyword_score": 0.0,
                    "final_score": 0.0,
                    "is_neighbor": True,
                }
            )

            seen.add(
                neighbor_key
            )

    # --------------------------------------------------------
    # Keep document order
    # --------------------------------------------------------

    expanded.sort(
        key=lambda item: (
            item["paper_id"],
            item["chunk_id"],
        )
    )

    return expanded


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    retrieved_chunks: list[dict],
    max_chunks: int = 8,
) -> str:

    if not retrieved_chunks:
        return ""

    retrieved_chunks = retrieved_chunks[
        :max_chunks
    ]

    context_parts = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        similarity = chunk.get(
            "similarity"
        )

        keyword = chunk.get(
            "keyword_score",
            0.0,
        )

        if similarity is None:

            similarity_text = "neighbor"

        else:

            similarity_text = (
                f"{similarity:.4f}"
            )

        context_parts.append(
            f"[SOURCE {index}]\n"
            f"Paper ID: {chunk['paper_id']}\n"
            f"Chunk ID: {chunk['chunk_id']}\n"
            f"Page: {chunk['page_number']}\n"
            f"Semantic Similarity: "
            f"{similarity_text}\n"
            f"Research Relevance: "
            f"{keyword:.4f}\n"
            f"Context Type: "
            f"{'NEIGHBOR' if chunk.get('is_neighbor') else 'RETRIEVED'}\n"
            f"Text:\n"
            f"{chunk['text']}\n"
        )

    return "\n".join(
        context_parts
    )