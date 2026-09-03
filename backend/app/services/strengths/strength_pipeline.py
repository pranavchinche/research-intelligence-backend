import json
import re

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.rag.retriever import (
    retrieve_chunks,
    build_context,
    expand_with_neighbors,
)

from app.services.llm import get_llm_answer


# ============================================================
# SYSTEM PROMPT
# ============================================================

STRENGTH_SYSTEM_PROMPT = """
You are a research-paper evidence extraction assistant.

Your task is to identify EXPLICIT, EVIDENCE-BACKED strengths of the
research paper from the supplied research-paper excerpts.

The supplied excerpts are the ONLY evidence available.

============================================================
CRITICAL RULE
============================================================

A strength MUST be explicitly supported by the paper text.

You must NOT invent:
- strengths that sound plausible but have no source evidence
- generic praise ("this is a well-written paper")
- strengths based on what the paper "could have" done well
- strengths derived from the paper merely existing or being published
- strengths about other papers or general research

============================================================
EVIDENCE RULE
============================================================

A statement is a strength ONLY when the source text contains
evidence such as:
- The authors explicitly describe a positive outcome, achievement,
  or advantage of their work
- The paper reports superior performance metrics (e.g. accuracy,
  BLEU, F1) compared to baselines or prior work
- The paper introduces a novel method, architecture, or technique
  that is described as effective or innovative
- The paper provides comprehensive evaluation, ablation studies,
  or analysis that validates their approach
- The paper uses a large-scale, diverse, or well-constructed
  dataset
- The paper demonstrates reproducibility (releasing code, data,
  or providing detailed experimental setup)
- The paper provides theoretical guarantees, proofs, or formal
  analysis
- The paper addresses a significant real-world problem with
  practical impact
- The paper combines existing techniques in a novel and effective way

If the source merely describes a method or parameter without
indicating it as a positive achievement, it is NOT a strength.

============================================================
CATEGORIES
============================================================

Each strength must be assigned to exactly one category:

- methodological: Novel or improved methods, algorithms,
  architectures, or frameworks
- experimental: Strong experimental design, comprehensive
  evaluation, thorough ablation studies, fair comparisons
- dataset: High-quality, large-scale, diverse, or novel datasets
  used or created
- contribution: Clear, significant contributions to the field
  or a specific problem
- theoretical: Theoretical analysis, proofs, formal guarantees,
  or mathematical foundations
- reproducibility: Code/data availability, detailed experimental
  setup, clear documentation enabling reproduction

============================================================
CITATIONS
============================================================

Every observed strength MUST contain a valid [SOURCE X]
citation through its "source" field.

Do not invent source numbers.

Use ONLY source numbers supplied in the context.

============================================================
QUANTITY
============================================================

Identify 3-7 strengths. Do NOT pad the output with weak or
marginal strengths. Only include strengths that have clear,
direct evidence in the source text.

If the paper genuinely has fewer than 3 clearly supported
strengths, return fewer. Quality over quantity.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

The JSON must have exactly this structure:

{
  "strengths": [
    {
      "statement": "Specific strength with evidence from the source",
      "category": "methodological",
      "source": "SOURCE X",
      "evidence": "Exact or near-exact quote from the source supporting this strength"
    }
  ]
}

If there are no explicitly supported strengths:

{
  "strengths": []
}
"""


# ============================================================
# BUILD STRENGTH PROMPT
# ============================================================

def build_strength_prompt(
    context: str,
) -> str:

    return f"""
{STRENGTH_SYSTEM_PROMPT}

============================================================
RESEARCH CONTEXT
============================================================

{context}

============================================================
TASK
============================================================

Read ALL supplied sources carefully.

Do not rely on one sentence alone when neighboring context
is available.

DO NOT copy or reuse any examples shown in this prompt.
Extract strengths ONLY from the actual supplied sources.

STEP 1
------

Read every source from SOURCE 1 through the last source.

For each source, look for sentences that explicitly describe
a positive achievement, advantage, superior performance,
novel contribution, strong design, or other clear strength
ABOUT THE AUTHORS' OWN WORK.

A statement IS a strength when the source text contains
evidence like: outperforms, achieves, state-of-the-art,
novel, effective, significant improvement, comprehensive,
thorough, large-scale, diverse, strong performance,
demonstrates, validates, guarantees, reproducible, released,
open-source.

A statement is NOT a strength when:
- The source merely describes a method without indicating
  it was effective or better than alternatives
- The source describes a standard procedure without
  indicating it was done particularly well
- There is no explicit positive assessment or result

STEP 2
------

For each identified strength:
- Assign it to exactly one category from:
  methodological, experimental, dataset, contribution,
  theoretical, reproducibility
- Write the strength statement grounded in the source
- Extract the supporting evidence (exact/near-exact quote)
- Cite the correct SOURCE X

STEP 3
------

Return ONLY valid JSON with this exact structure:

{{
  "strengths": [
    {{
      "statement": "Specific strength grounded in the source text",
      "category": "methodological",
      "source": "SOURCE X",
      "evidence": "Exact or near-exact quote from the source"
    }}
  ]
}}

If no explicit strength exists in ANY source:

{{
  "strengths": []
}}

============================================================
FINAL OUTPUT REQUIREMENT
============================================================

Return ONLY valid JSON.

Do not use Markdown.

Do not use ```json.

Do not add explanations before or after the JSON.

CRITICAL: Do NOT copy, repeat, or reuse any examples from this
prompt. The examples above are FICTIONAL ILLUSTRATIONS ONLY.
You must extract strengths ONLY from the actual text in the
supplied SOURCES. If a strength is not explicitly supported in
a source, do NOT invent it.
"""


# ============================================================
# HELPERS
# ============================================================

VALID_CATEGORIES = {
    "methodological",
    "experimental",
    "dataset",
    "contribution",
    "theoretical",
    "reproducibility",
}


def _extract_source_number(
    source: str,
) -> int | None:

    if not isinstance(source, str):
        return None

    match = re.fullmatch(
        r"\s*SOURCE\s+(\d+)\s*",
        source.strip(),
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return int(match.group(1))


def _normalize_source(
    source: str,
) -> str | None:

    number = _extract_source_number(source)

    if number is None:
        return None

    return f"SOURCE {number}"


_KEY_TERM_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on",
    "for", "with", "by", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "as",
    "at", "from", "which", "their", "such", "also", "than", "then",
    "not", "no", "do", "does", "did", "has", "have", "had", "can",
    "could", "should", "would", "may", "might", "will", "shall",
    "must", "into", "onto", "about", "over", "under", "between",
    "because", "due", "so", "if", "when", "while", "further",
    "more", "most", "some", "any", "each", "other", "only",
    "still", "yet", "however", "therefore", "thus",
}


def _extract_key_terms(
    text: str,
) -> set[str]:

    words = re.findall(
        r"[a-zA-Z]{4,}",
        text.lower(),
    )

    return {
        word
        for word in words
        if word not in _KEY_TERM_STOPWORDS
    }


def _statement_grounded_in_chunk(
    statement: str,
    chunk_text: str,
) -> bool:

    if not chunk_text or not chunk_text.strip():
        return False

    statement_terms = _extract_key_terms(statement)
    chunk_terms = _extract_key_terms(chunk_text)

    if not statement_terms or not chunk_terms:
        return True

    overlap = statement_terms & chunk_terms

    if len(statement_terms) <= 4:
        return len(overlap) >= 1

    return len(overlap) >= 2


def _resolve_source_chunk(
    source,
    source_chunks: list[dict] | None,
    requested_paper_id: int | None,
):

    if source is None:
        return None, None

    normalized_source = _normalize_source(
        str(source)
    )

    if not normalized_source:
        return None, None

    number = _extract_source_number(
        normalized_source
    )

    if number is None:
        return None, None

    if not source_chunks:
        return None, None

    if number < 1 or number > len(source_chunks):
        return None, None

    chunk = source_chunks[number - 1]

    if (
        requested_paper_id is not None
        and chunk.get("paper_id") != requested_paper_id
    ):
        return None, None

    return normalized_source, chunk


def _contains_any(
    text: str,
    patterns: list[str],
) -> bool:

    text = text.lower()

    return any(
        pattern.lower() in text
        for pattern in patterns
    )


# ============================================================
# GENERIC PRAISE REJECTION
# ============================================================

GENERIC_PRAISE_PATTERNS = [
    "this paper is well-written",
    "this is a well-written",
    "the paper is well-written",
    "well-written paper",
    "well written paper",
    "this paper makes a valuable contribution",
    "the paper makes a valuable",
    "important contribution to the field",
    "significant contribution to the field",
    "valuable contribution to the field",
    "this is a good paper",
    "this paper is good",
    "the paper is good",
    "this is a solid paper",
    "this paper is solid",
    "the paper is solid",
    "this is an excellent paper",
    "this paper is excellent",
    "the paper is excellent",
    "overall, the paper",
    "in general, the paper",
    "the paper is well-organized",
    "the paper is well structured",
    "the paper is well-written and",
    "comprehensive study",
    "comprehensive research",
    "interesting paper",
    "fascinating paper",
    "impressive work",
    "outstanding work",
    "great paper",
    "the writing is clear",
    "the presentation is clear",
    "the paper presents a clear",
]


# ============================================================
# VALIDATE LLM ANSWER
# ============================================================

def _no_strength_result() -> str:

    return json.dumps(
        {
            "strengths": [],
        },
        indent=2,
        ensure_ascii=False,
    )


def validate_strength_answer(
    answer: str,
    source_chunks: list[dict] | None = None,
    requested_paper_id: int | None = None,
) -> str:
    """
    Validate structured JSON returned by the LLM.

    Validation performs:
    1. JSON parsing.
    2. Root structure validation.
    3. SOURCE validation - citation exists, is in range, belongs
       to the requested paper_id.
    4. Groundedness validation - statement must share vocabulary
       with the cited chunk.
    5. Category validation.
    6. Generic-praise rejection.
    7. Evidence presence check.
    """

    if not answer or not answer.strip():
        return _no_strength_result()

    text = answer.strip()

    # ---------------------------------------------------------
    # Remove Markdown code fences
    # ---------------------------------------------------------

    if text.startswith("```"):

        lines = text.splitlines()

        if (
            lines
            and lines[0].strip().startswith("```")
        ):
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]

        text = "\n".join(
            lines
        ).strip()

    # ---------------------------------------------------------
    # Strip LLM preamble text before JSON
    # ---------------------------------------------------------

    json_start = text.find("{")
    json_end = text.rfind("}")

    if json_start != -1 and json_end > json_start:
        text = text[json_start : json_end + 1]

    # ---------------------------------------------------------
    # Parse JSON
    # ---------------------------------------------------------

    try:
        data = json.loads(text)
    except json.JSONDecodeError:

        print(
            "\nINVALID STRENGTH JSON FROM LLM"
        )
        print("=" * 80)
        print(text)

        return _no_strength_result()

    # ---------------------------------------------------------
    # Root validation
    # ---------------------------------------------------------

    if not isinstance(data, dict):
        return _no_strength_result()

    strengths = data.get("strengths", [])

    if not isinstance(strengths, list):
        strengths = []

    # =========================================================
    # VALIDATE STRENGTHS
    # =========================================================

    clean_strengths = []

    for item in strengths:

        if not isinstance(item, dict):
            continue

        statement = item.get("statement")
        category = item.get("category")
        source = item.get("source")
        evidence = item.get("evidence")

        if not statement:
            continue

        if not category:
            continue

        if not source:
            continue

        if not evidence:
            continue

        statement = str(statement).strip()
        category = str(category).strip().lower()
        evidence = str(evidence).strip()

        # -----------------------------------------------------
        # Validate category
        # -----------------------------------------------------

        if category not in VALID_CATEGORIES:
            continue

        # -----------------------------------------------------
        # Validate source citation
        # -----------------------------------------------------

        normalized_source, cited_chunk = _resolve_source_chunk(
            source,
            source_chunks,
            requested_paper_id,
        )

        if not normalized_source:

            print(
                "\nREJECTED STRENGTH:"
                " invalid, out-of-range, or cross-paper"
                " SOURCE citation"
            )
            print(f"Source: {source}")
            print(f"Statement: {statement}")

            continue

        # -----------------------------------------------------
        # Validate groundedness
        # -----------------------------------------------------

        if not _statement_grounded_in_chunk(
            statement,
            cited_chunk.get("text", ""),
        ):

            print(
                "\nREJECTED STRENGTH:"
                " statement is not grounded in the cited"
                " source text"
            )
            print(f"Source: {normalized_source}")
            print(f"Statement: {statement}")

            continue

        # -----------------------------------------------------
        # Reject generic praise
        # -----------------------------------------------------

        lower_statement = statement.lower()

        if _contains_any(
            lower_statement,
            GENERIC_PRAISE_PATTERNS,
        ):
            print(
                "\nREJECTED STRENGTH:"
                " generic praise detected"
            )
            print(statement)
            continue

        # -----------------------------------------------------
        # Validate evidence is grounded in chunk
        # -----------------------------------------------------

        if not _statement_grounded_in_chunk(
            evidence,
            cited_chunk.get("text", ""),
        ):

            print(
                "\nREJECTED STRENGTH:"
                " evidence is not grounded in the cited"
                " source text"
            )
            print(f"Source: {normalized_source}")
            print(f"Evidence: {evidence}")

            continue

        clean_strengths.append(
            {
                "statement": statement,
                "category": category,
                "source": normalized_source,
                "evidence": evidence,
            }
        )

    return json.dumps(
        {
            "strengths": clean_strengths,
        },
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# STRENGTH QUERIES
# ============================================================

STRENGTH_QUERIES = [
    "strengths of the proposed method",
    "advantages of the approach",
    "novel contributions of this work",
    "superior performance results",
    "state-of-the-art results",
    "outperforms baseline methods",
    "comprehensive evaluation and experiments",
    "ablation study results",
    "large-scale dataset used for evaluation",
    "theoretical analysis and guarantees",
    "code and data availability reproducibility",
    "effective method demonstrated",
    "significant improvement over existing methods",
    "innovative technique or architecture",
    "practical real-world impact",
]


# ============================================================
# STRENGTH EVIDENCE BONUS
# ============================================================

STRENGTH_EVIDENCE_BONUS = 0.05

STRENGTH_PATTERNS = [
    "outperforms",
    "outperformed",
    "state-of-the-art",
    "state of the art",
    "sota",
    "superior",
    "achieve",
    "achieved",
    "achieves",
    "significant improvement",
    "significant gain",
    "novel",
    "innovative",
    "effective",
    "efficient",
    "comprehensive",
    "thorough",
    "robust",
    "strong performance",
    "best performance",
    "best result",
    "best results",
    "highest accuracy",
    "highest score",
    "exceeds",
    "surpasses",
    "surpass",
    "demonstrate",
    "demonstrated",
    "validates",
    "validate",
    "ablation",
    "ablation study",
    "ablation studies",
    "large-scale",
    "diverse dataset",
    "real-world",
    "practical impact",
    "reproducible",
    "open-source",
    "open source",
    "released code",
    "code available",
    "theoretical guarantee",
    "theoretical analysis",
    "formal proof",
]


def _has_strength_evidence(
    text: str,
) -> bool:

    text_lower = text.lower()

    return any(
        pattern in text_lower
        for pattern in STRENGTH_PATTERNS
    )


def _candidate_sort_key(item: dict) -> float:

    base = item.get(
        "final_score",
        item.get("similarity", 0.0) or 0.0,
    )

    if base is None:
        base = 0.0

    bonus = (
        STRENGTH_EVIDENCE_BONUS
        if _has_strength_evidence(item.get("text", "") or "")
        else 0.0
    )

    return float(base) + bonus


# ============================================================
# RETRIEVAL CONSTANTS
# ============================================================

MIN_STRENGTH_FINAL_SCORE = 0.18

_REFERENCE_HEADINGS = (
    "references",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
    "author contributions",
    "conflict of interest",
    "conflicts of interest",
)


def _is_reference_chunk(text: str) -> bool:

    if not text:
        return False

    stripped = text.strip()

    if not stripped:
        return False

    lower = stripped.lower()

    first_line = lower.splitlines()[0].strip()

    if any(
        first_line.startswith(heading)
        or first_line == heading
        for heading in _REFERENCE_HEADINGS
    ):
        return True

    citation_markers = re.findall(
        r"\[\d{1,3}\]",
        stripped,
    )

    if len(citation_markers) >= 5:
        return True

    words = stripped.split()

    if words and (len(citation_markers) / max(len(words), 1)) > 0.05 and len(citation_markers) >= 3:
        return True

    return False


def _is_visualization_chunk(text: str) -> bool:

    if not text:
        return False

    stripped = text.strip()
    if not stripped:
        return False

    lower = stripped.lower()

    if lower.startswith("attention visualizations"):
        return True

    if "<pad>" in lower or "<eos>" in lower:
        return True

    lines = stripped.splitlines()
    if len(lines) >= 6:
        single_word_lines = sum(
            1 for line in lines
            if len(line.strip().split()) <= 1
            and len(line.strip()) > 0
        )
        if single_word_lines / max(len(lines), 1) > 0.5:
            return True

    return False


# ============================================================
# MAIN PIPELINE
# ============================================================

async def detect_strengths(
    db: AsyncSession,
    paper_id: int,
    top_k: int = 5,
):
    """
    Evidence-based strength detection pipeline.

    Follows the same grounding philosophy as gap detection:
    paper -> evidence retrieval -> LLM analysis -> validation
    -> structured response.
    """

    print(
        "\nSTRENGTH DETECTION PIPELINE"
    )
    print("=" * 80)
    print(f"Paper ID: {paper_id}")

    # ============================================================
    # 1. RETRIEVE EVIDENCE CHUNKS
    # ============================================================

    all_candidates = []

    for query in STRENGTH_QUERIES:

        results = await retrieve_chunks(
            db=db,
            query=query,
            paper_id=paper_id,
            top_k=top_k,
            min_similarity=0.20,
        )

        all_candidates.extend(results)

    print(
        f"\nTotal raw candidates: {len(all_candidates)}"
    )

    # ============================================================
    # 2. DEDUPLICATE
    # ============================================================

    seen = set()

    deduplicated = []

    for chunk in all_candidates:

        key = (
            chunk["paper_id"],
            chunk["chunk_id"],
        )

        if key in seen:
            continue

        seen.add(key)
        deduplicated.append(chunk)

    candidates = deduplicated

    print(
        f"After dedup: {len(candidates)}"
    )

    # ============================================================
    # 3. PAPER ISOLATION
    # ============================================================

    candidates = [
        c for c in candidates
        if c["paper_id"] == paper_id
    ]

    print(
        f"After paper isolation: {len(candidates)}"
    )

    # ============================================================
    # 4. FILTER REFERENCES AND VISUALIZATIONS
    # ============================================================

    filtered = []

    for chunk in candidates:

        text = chunk.get("text", "")

        if _is_reference_chunk(text):
            continue

        if _is_visualization_chunk(text):
            continue

        if chunk.get("final_score", 0.0) < MIN_STRENGTH_FINAL_SCORE:
            continue

        filtered.append(chunk)

    print(
        f"After filtering: {len(filtered)}"
    )

    # ============================================================
    # 5. SORT BY STRENGTH EVIDENCE
    # ============================================================

    filtered.sort(
        key=_candidate_sort_key,
        reverse=True,
    )

    # ============================================================
    # 6. SELECT TOP CANDIDATES
    # ============================================================

    selected = filtered[:10]

    print(
        f"Selected for context: {len(selected)}"
    )

    # ============================================================
    # 7. EXPAND WITH NEIGHBORS
    # ============================================================

    expanded_results = await expand_with_neighbors(
        db=db,
        retrieved_chunks=selected,
        window=1,
    )

    # ============================================================
    # 7.5 POST-EXPANSION CLEANUP
    # ============================================================

    final = []

    for chunk in expanded_results:

        if chunk["paper_id"] != paper_id:
            continue

        text = chunk.get("text", "")

        if _is_reference_chunk(text):
            continue

        if _is_visualization_chunk(text):
            continue

        final.append(chunk)

    # ============================================================
    # 8. SORT CONTEXT
    # ============================================================

    final.sort(
        key=lambda x: x.get(
            "final_score",
            x.get(
                "similarity",
                0.0,
            )
            or 0.0,
        ),
        reverse=True,
    )

    final = final[:15]

    print(
        "\nFINAL STRENGTH CONTEXT CHUNKS"
    )
    print("=" * 80)

    for chunk in final:

        context_type = (
            "NEIGHBOR"
            if chunk.get(
                "is_neighbor",
                False,
            )
            else "RETRIEVED"
        )

        print(
            f"Paper {chunk['paper_id']} | "
            f"Chunk {chunk['chunk_id']} | "
            f"{context_type}"
        )

    # ============================================================
    # 9. BUILD CONTEXT
    # ============================================================

    context = build_context(
        final,
        max_chunks=15,
    )

    if not context.strip():

        return {
            "paper_id": paper_id,
            "strengths": json.dumps(
                {
                    "strengths": [],
                },
                indent=2,
                ensure_ascii=False,
            ),
            "sources": final,
        }

    # ============================================================
    # 10. BUILD LLM PROMPT
    # ============================================================

    prompt = build_strength_prompt(
        context
    )

    # ============================================================
    # 11. GENERATE LLM ANSWER
    # ============================================================

    try:
        raw_answer = await get_llm_answer(
            prompt
        )
    except RuntimeError:
        raw_answer = '{"strengths":[]}'

    print(
        "\nRAW STRENGTH LLM OUTPUT"
    )
    print("=" * 80)
    print(raw_answer)

    # ============================================================
    # 12. VALIDATE LLM ANSWER
    # ============================================================

    answer = validate_strength_answer(
        raw_answer,
        source_chunks=final,
        requested_paper_id=paper_id,
    )

    # ============================================================
    # 13. FINAL RESULT
    # ============================================================

    return {
        "paper_id": paper_id,
        "strengths": answer,
        "sources": final,
    }
