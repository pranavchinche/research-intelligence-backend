# D:\FYP\main\backend\app\services\summary\summary_service.py

import json
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.paper import Paper
from app.models.paper_chunk import PaperChunk
from app.services.rag.retriever import (
    retrieve_chunks,
    build_context,
    expand_with_neighbors,
)
from app.services.llm import get_llm_answer
from app.services.llm.provider_registry import provider_registry


logger = logging.getLogger(__name__)


# ============================================================
# PROMPT
# ============================================================

SUMMARY_SYSTEM_PROMPT = """
You are a research-paper executive summary assistant.

Your task is to produce a structured executive summary of a
research paper using ONLY the supplied paper excerpts.

The supplied excerpts are the ONLY evidence available.

CRITICAL RULES:

1. Do NOT use outside knowledge.
2. Do NOT invent or infer information that is not explicitly
   present in the supplied text.
3. Every section MUST contain information grounded in the
   supplied sources.
4. If the paper does not provide enough evidence for a section,
   return exactly "Not stated in the paper." for that section.
5. Do NOT hallucinate methodology details, findings, or
   limitations that are not explicitly described.
6. Preserve the authors' own language and claims. Do not
   upgrade certainty (e.g. if the paper says "may", do not
   change it to "does").
7. Keep each section concise: 1-3 sentences maximum.
8. Do NOT invent metrics, numbers, or percentages that are
   not in the source text.
"""


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_summary_prompt(context: str) -> str:

    return f"""
{SUMMARY_SYSTEM_PROMPT}

============================================================
RESEARCH CONTEXT
============================================================

{context}

============================================================
TASK
============================================================

Read ALL supplied source excerpts carefully.

Produce a structured executive summary as a JSON object with
exactly these five keys:

{{
  "objective": "<1-3 sentences describing the paper's research
  objective or problem statement, grounded in the sources>",

  "main_contribution": "<1-3 sentences describing the paper's
  main contribution or proposed approach, grounded in the
  sources>",

  "methodology": "<1-3 sentences describing the methodology,
  experimental setup, or approach used, grounded in the
  sources>",

  "key_findings": "<1-3 sentences summarizing the key results
  or findings, grounded in the sources>",

  "limitations_conclusion": "<1-3 sentences covering stated
  limitations, future work, or conclusions, grounded in the
  sources>"
}}

RULES FOR EACH KEY:

- If the supplied context does not contain enough evidence for
  a key, set its value to exactly: Not stated in the paper.
- Do NOT add extra keys.
- Do NOT add explanations outside the JSON.
- Do NOT add markdown code fences around the JSON.
- Output ONLY the raw JSON object.

============================================================
OUTPUT
============================================================
"""


# ============================================================
# DEFAULT VALUES
# ============================================================

NOT_STATED = "Not stated in the paper."

# Canonical output keys (the shape the frontend expects).
VALID_KEYS = {
    "objective",
    "main_contribution",
    "methodology",
    "key_findings",
    "limitations_conclusion",
}

# Aliases the LLM may use instead of the canonical keys.
# Each canonical key maps to a set of acceptable spellings.
KEY_ALIASES = {
    "objective": {"objective", "research_objective", "goal", "problem_statement"},
    "main_contribution": {
        "main_contribution",
        "contribution",
        "main_contributions",
        "key_contribution",
        "proposed_approach",
        "contributions",
    },
    "methodology": {"methodology", "methods", "method", "approach", "experimental_setup"},
    "key_findings": {
        "key_findings",
        "findings",
        "key_finding",
        "results",
        "key_results",
        "main_findings",
    },
    "limitations_conclusion": {
        "limitations_conclusion",
        "limitations_and_conclusion",
        "limitations",
        "conclusion",
        "conclusions",
        "limitations_&_conclusion",
        "limitations_conclusions",
    },
}


def _deep_extract(data) -> dict | None:
    """
    Recursively search for a dict that contains the expected
    summary keys (matching by canonical name OR alias).
    Handles:
      - a plain dict with the five keys
      - a JSON string that parses to such a dict
      - a dict whose values are stringified JSON containing
        the five keys
      - a dict wrapping the summary under another key
    Returns the matching dict or None.
    """
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
        return _deep_extract(parsed)

    if isinstance(data, dict):
        if _has_summary_keys(data):
            return data
        for value in data.values():
            if isinstance(value, str) and value.strip():
                result = _deep_extract(value)
                if result is not None:
                    return result
            elif isinstance(value, dict):
                result = _deep_extract(value)
                if result is not None:
                    return result
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, (dict, str)):
                        result = _deep_extract(item)
                        if result is not None:
                            return result

    return None


def _has_summary_keys(data: dict) -> bool:
    """
    True if the dict contains (via canonical name or alias) at
    least the core fields.  The canonical name for each of the
    five sections is checked against the dict keys; alias names
    also count.
    """
    flat_keys = {str(k).strip().lower() for k in data.keys()}
    hits = 0
    for canonical, aliases in KEY_ALIASES.items():
        if any(alias in flat_keys for alias in aliases):
            hits += 1
    # Require at least the majority of sections to be present so we
    # don't accidentally accept unrelated dicts.
    return hits >= 3


def _normalize_key(raw_key: str) -> str | None:
    """
    Map a raw key (as returned by the LLM) to the canonical
    key name, or None if unrecognized.
    """
    normalized = str(raw_key).strip().lower().replace(" ", "_")
    for canonical, aliases in KEY_ALIASES.items():
        if normalized in aliases:
            return canonical
        # Also allow fuzzy match where an alias is a substring
        # (e.g. "research_objectives").
        for alias in aliases:
            if normalized.startswith(alias) or alias.startswith(normalized):
                if normalized in {alias, alias + "s"} or alias in {normalized, normalized + "s"}:
                    return canonical
    return None


def _try_parse_json(candidate: str):
    """
    Attempt to parse a JSON candidate string with increasing
    tolerance (strip trailing commas, trailing prose, repair
    truncated/missing-brace JSON, strip markdown fences).
    Returns a parsed value or None.
    """
    if not candidate:
        return None

    # 1. Direct parse.
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences.
    cleaned = candidate.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return _try_parse_json(cleaned)

    # 3. Strip a trailing comma before the final closing brace
    #    (e.g. '...,"key": "value",}').
    stripped = cleaned.rstrip()
    if stripped.endswith(",}"):
        fixed = stripped[:-2] + "}"
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    # 4. Repair a truncated JSON object missing its closing
    #    brace(s).  LLMs frequently hit their token limit while
    #    emitting JSON and the final '}' is dropped.  If the text
    #    starts with '{' and the last non-empty character is a
    #    '"', ']', or ',' (i.e. a closing brace was cut off), try
    #    appending the missing closing brace.
    if cleaned.startswith("{") and not cleaned.endswith("}"):
        repaired = cleaned + "}"
        try:
            parsed = json.loads(repaired)
            return parsed
        except json.JSONDecodeError:
            # The last value may itself be truncated (unterminated
            # string) — too broken to repair reliably.
            pass

    return None


def _extract_json_object(text: str) -> dict | None:
    """
    Try to extract a JSON object from text that has surrounding
    non-JSON prose (e.g. "Here is the summary: {...}").

    Searches for JSON objects, trying the largest valid parse.
    Handles:
      - prose before/after the JSON
      - multiple JSON blocks (uses the one with summary keys)
      - trailing commas
      - first '{' ... last '}' approach as a fallback
    """
    if not text or not text.strip():
        return None

    data = None

    # Strategy 1: try parsing the whole thing (this also triggers
    # the repair path for truncated JSON).
    data = _try_parse_json(text)
    if data is not None:
        return data

    # Strategy 2: find every balanced {...} block and try each.
    candidates = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidates.append(text[start : i + 1])
                start = -1
    # If unbalanced, fall back to the whole-range approach.
    if not candidates:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            candidates = [text[start : end + 1]]
        elif start != -1:
            # No closing brace at all: treat rest of text as the
            # (possibly truncated) object.
            candidates = [text[start:]]

    # Try each candidate, longest-first (most likely the full
    # JSON object), with trailing-comma / truncation tolerance.
    candidates.sort(key=len, reverse=True)
    for candidate in candidates:
        parsed = _try_parse_json(candidate)
        if parsed is not None:
            data = parsed
            break

    return data


def _validate_summary(raw: str) -> dict | None:
    """
    Parse and validate the LLM JSON output.
    Returns a clean dict (with canonical keys) or None on
    total failure.
    """
    if not raw or not str(raw).strip():
        return None

    text = str(raw).strip()

    logger.info(
        "Summary raw parse input: len=%d, prefix=%r",
        len(text),
        text[:120],
    )

    # Attempt direct JSON parse.
    data = _try_parse_json(text)

    # If direct parse failed, try extracting a JSON object from
    # surrounding prose / multiple blocks.
    if data is None:
        data = _extract_json_object(text)

    if data is None:
        logger.warning(
            "Summary: no JSON could be extracted from raw output."
        )
        return None

    # Attempt deep extraction to handle nested/stringified JSON.
    extracted = _deep_extract(data)
    if extracted is not None:
        data = extracted

    if not isinstance(data, dict):
        logger.warning(
            "Summary: extracted data is not a dict (type=%s).",
            type(data).__name__,
        )
        return None

    # Map keys by canonical name or alias.
    validated: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(value, str):
            continue
        value = value.strip()
        if not value:
            continue
        canonical = _normalize_key(key)
        if canonical and canonical not in validated:
            validated[canonical] = value

    # If nothing mapped, fail.
    if not validated:
        logger.warning(
            "Summary: no recognized summary keys found. keys=%r",
            [str(k) for k in data.keys()],
        )
        return None

    # Fill any missing sections with "Not stated in the paper."
    for canonical in VALID_KEYS:
        if canonical not in validated:
            validated[canonical] = NOT_STATED

    return validated


# ============================================================
# MAIN SERVICE
# ============================================================

async def generate_executive_summary(
    db: AsyncSession,
    paper_id: int,
) -> dict:
    """
    Generate an AI executive summary for a paper.

    Follows the same pipeline pattern as gap_pipeline.py:
    1. Fetch paper from DB
    2. Retrieve relevant chunks via RAG
    3. Build context
    4. Generate via LLM
    5. Validate output
    """

    # ========================================================
    # 1. FETCH PAPER
    # ========================================================

    stmt = select(Paper).where(Paper.id == paper_id)
    result = await db.execute(stmt)
    paper = result.scalar_one_or_none()

    if paper is None:
        return {
            "paper_id": paper_id,
            "error": "Paper not found",
            "summary": None,
            "sources": [],
        }

    # ========================================================
    # 2. CHECK FOR FULL TEXT / CHUNKS
    # ========================================================

    has_full_text = bool(
        paper.full_text
        and paper.full_text.strip()
    )

    has_chunks_stmt = (
        select(PaperChunk)
        .where(PaperChunk.paper_id == paper_id)
        .limit(1)
    )
    chunks_result = await db.execute(has_chunks_stmt)
    has_chunks = (
        chunks_result.scalar_one_or_none() is not None
    )

    if not has_full_text and not has_chunks:
        return {
            "paper_id": paper_id,
            "summary": {
                "objective": (
                    "No text content available for this paper. "
                    "The paper may not have been processed yet."
                ),
                "main_contribution": NOT_STATED,
                "methodology": NOT_STATED,
                "key_findings": NOT_STATED,
                "limitations_conclusion": NOT_STATED,
            },
            "sources": [],
        }

    # ========================================================
    # 3. BUILD CONTEXT
    # ========================================================

    source_chunks: list[dict] = []
    context = ""

    if has_chunks:

        # Use a broad query to get coverage across the
        # entire paper.
        broad_query = (
            f"{paper.title or ''} "
            f"{paper.abstract or ''} "
            "objective methodology results findings "
            "conclusion limitations future work"
        ).strip()

        logger.info(
            "Summary retrieval: paper_id=%s, query_len=%d",
            paper_id,
            len(broad_query),
        )

        results = await retrieve_chunks(
            db=db,
            query=broad_query,
            paper_id=paper_id,
            top_k=15,
            min_similarity=0.05,
        )

        logger.info(
            "Summary retrieval: chunks_found=%d",
            len(results),
        )

        if results:

            expanded_results = (
                await expand_with_neighbors(
                    db=db,
                    retrieved_chunks=results,
                    window=1,
                )
            )

            context = build_context(
                expanded_results,
                max_chunks=20,
            )

            source_chunks = expanded_results

    # Fallback: use full_text directly if RAG yielded
    # nothing but full_text exists.
    if not context.strip() and has_full_text:

        truncated = paper.full_text[:12000]

        context = (
            "[SOURCE 1]\n"
            f"Paper ID: {paper_id}\n"
            "Full Text (truncated):\n"
            f"{truncated}"
        )

    if not context.strip():
        return {
            "paper_id": paper_id,
            "summary": {
                "objective": NOT_STATED,
                "main_contribution": NOT_STATED,
                "methodology": NOT_STATED,
                "key_findings": NOT_STATED,
                "limitations_conclusion": NOT_STATED,
            },
            "sources": [],
        }

    logger.info(
        "Summary context: paper_id=%s, "
        "abstract_len=%d, full_text_len=%d, "
        "chunks=%d, context_len=%d",
        paper_id,
        len(paper.abstract or ""),
        len(paper.full_text or ""),
        len(source_chunks),
        len(context),
    )

    # ========================================================
    # 4. BUILD PROMPT
    # ========================================================

    prompt = build_summary_prompt(context)

    # ========================================================
    # 5. GENERATE LLM ANSWER
    # ========================================================

    try:
        raw_answer = await get_llm_answer(prompt)
    except RuntimeError as exc:
        logger.warning(
            "Summary LLM generation failed for paper %s: %s",
            paper_id,
            exc,
        )
        return {
            "paper_id": paper_id,
            "error": (
                "LLM generation failed: all providers "
                "are unavailable."
            ),
            "summary": None,
            "sources": source_chunks,
        }

    logger.info(
        "Summary LLM output: paper_id=%s, "
        "provider=%s, provider_response_len=%d",
        paper_id,
        provider_registry.last_used_provider,
        len(raw_answer),
    )
    # Bounded preview (never log full large payloads).
    logger.debug(
        "Summary raw LLM output preview: %r",
        raw_answer[:500],
    )

    # ========================================================
    # 6. VALIDATE
    # ========================================================

    validated = _validate_summary(raw_answer)

    if validated is None:

        logger.warning(
            "Summary validation failed for paper %s. "
            "Raw output (first 200 chars): %r",
            paper_id,
            raw_answer[:200],
        )

        return {
            "paper_id": paper_id,
            "error": (
                "LLM returned a response that could not "
                "be parsed as a structured summary."
            ),
            "summary": None,
            "sources": source_chunks,
        }

    logger.info(
        "Summary validated OK for paper %s: "
        "keys=%s",
        paper_id,
        list(validated.keys()),
    )

    # ========================================================
    # 7. FINAL RESULT
    # ========================================================

    return {
        "paper_id": paper_id,
        "summary": validated,
        "sources": source_chunks,
    }
