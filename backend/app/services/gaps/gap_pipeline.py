# D:\FYP\main\backend\app\services\gaps\gap_pipeline.py

import json
import re

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.rag.retriever import (
    retrieve_chunks,
    build_context,
    expand_with_neighbors,
    has_limitation_evidence,
)

from app.services.llm import get_llm_answer

from app.services.gaps.limitation_verifier import verify_limitation


async def _verify_limitations(
    answer: str,
    source_chunks: list[dict],
    requested_paper_id: int | None,
) -> str:
    """
    Attach a per-limitation LLM verification result to the validated
    gap JSON. Additive only: each limitation item gains a
    "verification" object ({valid, evidence}); the overall response
    shape (limitations/possible_gaps) is unchanged. Any verification
    failure degrades to leaving items unannotated.
    """

    try:
        parsed = json.loads(answer)
    except (ValueError, TypeError):
        return answer

    limitations = parsed.get("limitations")

    if not isinstance(limitations, list) or not limitations:
        return answer

    for item in limitations:
        if not isinstance(item, dict):
            continue

        _, chunk = _resolve_source_chunk(
            item.get("source"),
            source_chunks,
            requested_paper_id,
        )

        chunk_text = (
            chunk.get("text", "")
            if isinstance(chunk, dict)
            else ""
        )

        if not chunk_text.strip():
            continue

        try:
            verification = await verify_limitation(chunk_text)
        except Exception:
            continue

        item["verification"] = {
            "valid": bool(verification.get("valid")),
            "evidence": verification.get("evidence"),
        }

    return json.dumps(
        parsed,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

GAP_SYSTEM_PROMPT = """
You are a research-paper evidence extraction assistant.

Your task is to identify EXPLICIT limitations and CONSERVATIVE
possible research gaps from the supplied research-paper excerpts.

The supplied excerpts are the ONLY evidence available.

============================================================
CRITICAL RULE
============================================================

A possible research gap is NOT a solution.

You must NOT invent:

- methods
- algorithms
- technologies
- datasets
- experiments
- applications
- architectures
- optimization techniques
- models
- solutions

You may ONLY derive a possible gap from something explicitly
identified in the supplied text as:

- a limitation
- a weakness
- a difficulty
- a challenge
- a trade-off
- performance degradation
- an unresolved issue
- a stated need for improvement

============================================================
EVIDENCE RULE
============================================================

A statement is NOT a limitation merely because:

- the paper does not mention something
- another technique exists
- the paper uses a particular parameter
- the paper uses an older technique
- the paper does not test something
- the paper could theoretically be improved

NOT MENTIONED does NOT mean UNRESOLVED.

If the supplied context does not explicitly establish
a limitation, do NOT create one.

============================================================
NEIGHBOR EVIDENCE RULE
============================================================

Some sources are marked:

Context Type: RETRIEVED

Others are marked:

Context Type: NEIGHBOR

NEIGHBOR chunks are included because surrounding text may
contain evidence showing that an apparent limitation was
actually addressed by the authors.

Therefore:

DO NOT classify a problem as unresolved until you inspect
the surrounding evidence supplied in the context.

Example:

Chunk A:
"LSTMs can have exploding gradients."

Chunk B:
"We enforced a hard constraint on the norm of the gradient."

Correct conclusion:

The exploding-gradient problem was addressed.

Therefore it is NOT an unresolved research gap.

============================================================
ADDRESSING STATUS
============================================================

For every candidate limitation, determine exactly one status.

Allowed statuses:

- ADDRESSED
- PARTIALLY ADDRESSED
- EXPLICITLY UNRESOLVED
- INSUFFICIENT EVIDENCE

Only:

- PARTIALLY ADDRESSED
- EXPLICITLY UNRESOLVED

may produce a possible research gap.

Do NOT generate a gap for:

- ADDRESSED
- INSUFFICIENT EVIDENCE

============================================================
DEFAULT: PRODUCE A GAP FOR EVERY ACCEPTED LIMITATION
============================================================

For every limitation whose status is PARTIALLY ADDRESSED or
EXPLICITLY UNRESOLVED, you MUST normally produce exactly one
corresponding possible_gaps entry describing THE SAME
underlying problem.

The possible gap is a conservative RESTATEMENT of the
limitation, not a new observation. It must:

- reuse the same terminology and subject matter as the
  limitation (e.g. if the limitation says "sample", the gap
  must talk about the sample; if it says "generalizability",
  the gap must talk about generalizability),
- remain a PROBLEM STATEMENT describing what is unresolved,
  never an instruction, recommendation, or action.

Turning a limitation into a gap means rephrasing it as an
open problem, for example:

Limitation:
"The sample consists of 92 students from one university and
is not representative."

Correct possible gap:
"The representativeness of the sample remains limited,
restricting generalizability."

Incorrect possible gap (this is a solution, reject it):
"Collect a more representative sample from multiple
universities."

Do NOT skip producing a gap just because the limitation
already sounds complete, repetitive, or obvious. Repetition
of the underlying problem, in different words, is exactly
what is required here.

Only omit a possible_gaps entry for an accepted limitation if
producing one would require you to guess, infer, or add a
detail that is not explicitly present in the supplied source
text for that limitation. In that case it is acceptable to
leave that limitation without a gap rather than invent one -
but this should be the exception, not the default.

============================================================
IMPORTANT GAP RELATIONSHIP
============================================================

Every possible research gap MUST directly originate from
one limitation.

The source of the possible gap MUST be the same SOURCE X
as the limitation from which it was derived.

Do NOT create a possible gap from a source that does not
contain an accepted limitation.

============================================================
TECHNICAL DETAILS
============================================================

Technical details are NOT automatically limitations.

For example:

"The model uses dropout with Pdrop = 0.1."

This alone is NOT a limitation.

However:

"The model uses dropout with Pdrop = 0.1,
which reduces performance under condition X."

This can be a limitation because the source explicitly
describes a negative effect.

============================================================
CITATIONS
============================================================

Every observed limitation MUST contain a valid [SOURCE X]
citation through its "source" field.

Every possible research gap MUST use the same source
as the limitation from which it was derived.

Do not invent source numbers.

Use ONLY source numbers supplied in the context.

============================================================
NO NOVELTY CLAIMS
============================================================

Never claim:

- novel
- unprecedented
- first
- unexplored
- no previous work exists
- nobody has solved this
- globally unexplored

You are identifying possible research opportunities,
not proving novelty.

============================================================
NO SOLUTIONS
============================================================

Do NOT write:

"Therefore, a new Transformer architecture should be developed."

Do NOT write:

"Reinforcement learning could solve this."

Do NOT write:

"The authors should use method X."

The gap must remain at the PROBLEM level.

============================================================
GAP RULE
============================================================

A possible research gap must describe ONLY the unresolved
research problem.

The gap must be a conservative restatement of the limitation.

IMPORTANT:
Never phrase the gap as an action or solution. Do not begin the gap
with or recommend actions such as developing, designing, creating,
building, improving, increasing, expanding, collecting, adopting,
applying, implementing, using, investigating, evaluating, testing,
or exploring something.

BAD:
"Developing a more representative sample could improve generalizability."

GOOD:
"The representativeness of the sample remains limited, restricting
generalizability."

Do NOT introduce:

- a new method
- a new algorithm
- a new architecture
- a new technology
- a new dataset
- a new experiment
- a new model
- a proposed solution

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

The JSON must have exactly this structure:

{
  "limitations": [
    {
      "statement": "Explicit limitation from the source",
      "status": "PARTIALLY ADDRESSED",
      "source": "SOURCE X"
    }
  ],
  "possible_gaps": [
    {
      "statement": "Conservative research gap describing the same unresolved problem",
      "source": "SOURCE X"
    }
  ]
}

If there is no explicit limitation:

{
  "limitations": [],
  "possible_gaps": []
}

If limitations exist but all are addressed:

{
  "limitations": [],
  "possible_gaps": []
}
"""


# ============================================================
# BUILD GAP PROMPT
# ============================================================

def build_gap_prompt(
    context: str,
) -> str:

    return f"""
{GAP_SYSTEM_PROMPT}

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
Extract limitations ONLY from the actual supplied sources.

STEP 1
------

Read every source from SOURCE 1 through the last source.

For each source, look for sentences that explicitly state
a negative effect, limitation, weakness, trade-off,
difficulty, or unresolved issue ABOUT THE AUTHORS' OWN
METHOD OR MODEL.

A statement IS a limitation when the source text contains
words like: worse, poorer, degrades, hurts, limited,
cannot, fails, difficulty, challenge, downside, trade-off,
compromise, drawback, restriction, constrains, prevents,
underperforms, remains an open problem, is not sufficient.

A statement is NOT a limitation when:
- The source just describes a method, parameter, or result
  without saying it causes a problem
- The paper does not mention something (absence is not a
  limitation unless the authors explicitly call it out)
- The source states a positive result or successful outcome

The key question for EACH candidate: Does the source text
EXPLICITLY describe this as a problem, weakness, degradation,
trade-off, difficulty, or unresolved issue?

If the source just describes a method, parameter, result, or
technical detail WITHOUT saying it causes a problem, it is
NOT a limitation.

STEP 2
------

For each identified limitation, check neighboring sources
to determine whether the authors addressed it.

Statuses:
- ADDRESSED: the authors explicitly solve this problem
- PARTIALLY ADDRESSED: the authors mitigate but do not
  fully solve it
- EXPLICITLY UNRESOLVED: the authors acknowledge the
  problem remains
- INSUFFICIENT EVIDENCE: unclear from the text

STEP 3
------

For each PARTIALLY ADDRESSED or EXPLICITLY UNRESOLVED
limitation, create one possible_gaps entry that
conservatively restates the same problem. You MUST produce
a possible_gaps entry for every PARTIALLY ADDRESSED or
EXPLICITLY UNRESOLVED limitation.

STEP 4
------

Return ONLY valid JSON with this exact structure:

{{
  "limitations": [
    {{
      "statement": "The specific limitation as stated in the source",
      "status": "PARTIALLY ADDRESSED",
      "source": "SOURCE X"
    }}
  ],
  "possible_gaps": [
    {{
      "statement": "Conservative restatement of the same unresolved problem",
      "source": "SOURCE X"
    }}
  ]
}}

If no explicit limitation exists in ANY source:

{{
  "limitations": [],
  "possible_gaps": []
}}

============================================================
FINAL OUTPUT REQUIREMENT
============================================================

Return ONLY valid JSON.

Do not use Markdown.

Do not use ```json.

Do not add explanations before or after the JSON.

CRITICAL: Do NOT copy, repeat, or reuse any examples from this
prompt. The examples above (including any mention of "512 tokens",
"A100 GPUs", "Label smoothing", or "dropout rate") are FICTIONAL
ILLUSTRATIONS ONLY. You must extract limitations ONLY from the
actual text in the supplied SOURCES. If a limitation is not
explicitly stated in a source, do NOT invent it.
"""


# ============================================================
# HELPERS
# ============================================================

VALID_STATUSES = {
    "ADDRESSED",
    "PARTIALLY ADDRESSED",
    "EXPLICITLY UNRESOLVED",
    "INSUFFICIENT EVIDENCE",
}


# ============================================================
# RETRIEVAL QUALITY CONSTANTS
# ============================================================

# A candidate chunk must clear this hybrid (semantic + keyword)
# score to be considered as gap evidence at all. This is on top
# of the per-query min_similarity already applied in
# retrieve_chunks() - it protects against chunks that only
# scraped past that bar because of the keyword bonus (e.g. a
# references section that happens to contain "LSTM" or
# "network" in a paper title).
MIN_GAP_FINAL_SCORE = 0.18

# Chunks that already look like they discuss a limitation get a
# small priority boost when ranking candidates, so genuinely
# relevant evidence is preferred over lexically-similar but
# off-topic chunks.
LIMITATION_EVIDENCE_BONUS = 0.05

# Headings/markers that indicate a chunk is bibliography-like
# rather than paper body text.
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
    """
    Heuristically detect bibliography/references/acknowledgments
    chunks so they are never used as gap evidence.

    This is intentionally conservative: it only flags chunks that
    clearly look like reference-list material, either because they
    open with a references-type heading or because they are
    dominated by bracketed numeric citation markers (e.g. "[12]",
    "[3]"), which is the hallmark of a reference list entry rather
    than paper body prose.
    """

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

    # A dense run of bracketed numeric markers relative to the
    # chunk's length is characteristic of a reference list, not
    # of normal prose (which cites sparingly).
    if len(citation_markers) >= 5:
        return True

    words = stripped.split()

    if words and (len(citation_markers) / max(len(words), 1)) > 0.05 and len(citation_markers) >= 3:
        return True

    return False


def _is_visualization_chunk(text: str) -> bool:
    """
    Detect chunks that contain attention visualization output
    or similar non-body-text content (individual words on
    separate lines, padding tokens, etc.). These are appendix
    artifacts, not paper body prose.
    """

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


def _candidate_sort_key(item: dict) -> float:
    """
    Ranking key used to select gap candidates: hybrid score, with
    a small boost for chunks that already look like they discuss
    a limitation.
    """

    base = item.get(
        "final_score",
        item.get("similarity", 0.0) or 0.0,
    )

    if base is None:
        base = 0.0

    bonus = (
        LIMITATION_EVIDENCE_BONUS
        if has_limitation_evidence(item.get("text", "") or "")
        else 0.0
    )

    return float(base) + bonus


def _extract_source_number(
    source: str,
) -> int | None:
    """
    Extract SOURCE number from strings such as:

        SOURCE 1
        SOURCE 10
        source 3

    Returns None for invalid source strings.
    """

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
    """
    Normalize source citation to:

        SOURCE X
    """

    number = _extract_source_number(source)

    if number is None:
        return None

    return f"SOURCE {number}"


def _contains_any(
    text: str,
    patterns: list[str],
) -> bool:

    text = text.lower()

    return any(
        pattern.lower() in text
        for pattern in patterns
    )


def _looks_like_solution_statement(
    statement: str,
) -> bool:
    """
    Reject common action-oriented solution statements.

    The gap must remain at the unresolved-problem level, not become
    an instruction to develop, design, improve, collect, or implement
    something.
    """

    text = statement.strip().lower()

    action_starts = (
        "develop ",
        "developing ",
        "design ",
        "designing ",
        "create ",
        "creating ",
        "build ",
        "building ",
        "improve ",
        "improving ",
        "increase ",
        "increasing ",
        "expand ",
        "expanding ",
        "collect ",
        "collecting ",
        "adopt ",
        "adopting ",
        "apply ",
        "applying ",
        "implement ",
        "implementing ",
        "use ",
        "using ",
        "integrate ",
        "integrating ",
        "introduce ",
        "introducing ",
        "evaluate ",
        "evaluating ",
        "test ",
        "testing ",
        "investigate ",
        "investigating ",
        "explore ",
        "exploring ",
    )

    return text.startswith(action_starts)


# Common words that carry no distinguishing topical meaning.
# Used only to filter noise out of the key-term overlap check
# below - this is NOT a rejection pattern list, so common
# words like "limited", "limitation", "difficult", "challenge",
# "restrict", "degradation", "trade-off", and "remains" are
# intentionally treated as ordinary content terms and are kept
# (not filtered), since they are exactly the kind of vocabulary
# a valid problem statement legitimately shares with its source
# limitation.

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
    """
    Extract lowercase content words (length >= 4) from a
    statement, for use in a conservative overlap check only.

    This deliberately does NOT strip topical words such as
    "limited", "limitation", "restrict", "challenge",
    "degradation", or "remains" - those are meaningful content
    terms, not noise.
    """

    words = re.findall(
        r"[a-zA-Z]{4,}",
        text.lower(),
    )

    return {
        word
        for word in words
        if word not in _KEY_TERM_STOPWORDS
    }


def _gap_shares_key_terms(
    gap_statement: str,
    limitation_statements: list[str],
) -> bool:
    """
    Conservative relatedness check between a possible gap and
    the limitation(s) it claims to come from.

    This only catches the CLEARLY unrelated case (zero shared
    content words with every candidate limitation for that
    source). It never requires word-for-word equality, and it
    fails open (returns True) whenever there isn't enough
    signal to safely judge - to avoid false-positive rejections
    of valid, differently-worded problem statements.
    """

    if not limitation_statements:
        return True

    gap_terms = _extract_key_terms(
        gap_statement
    )

    if not gap_terms:
        return True

    for limitation_statement in limitation_statements:

        limitation_terms = _extract_key_terms(
            limitation_statement
        )

        if gap_terms & limitation_terms:
            return True

    return False


def _statement_grounded_in_chunk(
    statement: str,
    chunk_text: str,
) -> bool:
    """
    Conservative groundedness check between an LLM-produced
    statement and the actual text of the source chunk it claims
    to be citing.

    This only rejects the clear hallucination case: a statement
    that shares essentially no content vocabulary with the chunk
    it is supposedly drawn from (e.g. a statement about "weather
    conditions and travel modes" attributed to a chunk that never
    mentions either). It never requires exact wording or a full
    paraphrase match, and it fails open when there isn't enough
    signal to safely judge, so genuinely-supported statements that
    are simply reworded are not penalized.
    """

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
    """
    Resolve a raw "source" field (e.g. "SOURCE 3") to the actual
    context chunk it refers to.

    Returns (normalized_source, chunk) on success, or (None, None)
    if the citation is missing, malformed, out of range for the
    supplied context, or points at a chunk from a different paper
    than the one requested.
    """

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


# ============================================================
# VALIDATION PATTERNS
# ============================================================

# These indicate that the LLM has moved from describing
# a research problem into proposing a solution.

FORBIDDEN_SOLUTION_PATTERNS = [

    "should use",
    "should adopt",
    "should apply",
    "should implement",

    "could use",
    "could adopt",
    "could apply",
    "could implement",

    "can use",
    "can adopt",
    "can apply",
    "can implement",

    "could support",
    "should support",
    "can support",

    "could solve",
    "should solve",
    "can solve",

    "could address",
    "should address",
    "can address",

    "could improve",
    "should improve",
    "can improve",

    "could mitigate",
    "should mitigate",
    "can mitigate",

    "could reduce",
    "should reduce",
    "can reduce",

    "could increase",
    "should increase",
    "can increase",

    "could develop",
    "should develop",
    "can develop",

    "could be developed",
    "should be developed",
    "can be developed",

    "develop a new",
    "developing a new",

    # Action-oriented solution constructions
    "develop a more",
    "developing a more",
    "develop a better",
    "developing a better",
    "develop an improved",
    "developing an improved",

    "design a more",
    "designing a more",
    "design a better",
    "designing a better",
    "design an improved",
    "designing an improved",

    "create a more",
    "creating a more",
    "create a better",
    "creating a better",
    "create an improved",
    "creating an improved",

    "improve the",
    "improving the",
    "increase the",
    "increasing the",
    "expand the",
    "expanding the",

    "collect more",
    "collecting more",
    "use a more",
    "using a more",
    "use a better",
    "using a better",
    "use an improved",
    "using an improved",

    "adopt a more",
    "adopting a more",
    "apply a more",
    "applying a more",
    "implement a more",
    "implementing a more",

    "develop a representative",
    "developing a representative",
    "create a representative",
    "creating a representative",
    "design a representative",
    "designing a representative",

    "develop a more representative",
    "developing a more representative",
    "create a more representative",
    "creating a more representative",
    "design a more representative",
    "designing a more representative",

    "design a new",
    "designing a new",

    "propose a method",
    "propose a new method",

    "propose an architecture",
    "propose a new architecture",

    "introduce a new method",
    "introduce a new architecture",
    "introduce a new algorithm",

    "new architecture",
    "new method",
    "new algorithm",
    "new model",
    "new framework",
    "new technique",

    "reinforcement learning could",
    "using reinforcement learning",

    "using more gpu",
    "using more gpus",
    "using fewer gpu",
    "using fewer gpus",

    "use transformer",
    "use transformers",
    "using transformer",
    "using transformers",

    "use attention",
    "using attention",

    "use lstm",
    "using lstm",

    "use cnn",
    "using cnn",

    "use rnn",
    "using rnn",

    "use a larger model",
    "use a smaller model",
    "use more data",
    "collect more data",

    "future work should",
    "future work could",

    "should be validated",
    "could be validated",

    "should be evaluated",
    "could be evaluated",

    "should be tested",
    "could be tested",

    "should be investigated",
    "could be investigated",

    "should be explored",
    "could be explored",
]


# These usually indicate that the model is claiming a gap
# merely because the paper failed to mention/test something.

UNSUPPORTED_ABSENCE_PATTERNS = [

    "did not discuss",
    "does not discuss",
    "not discussed",

    "did not address",
    "does not address",
    "not addressed",

    "did not test",
    "does not test",
    "not tested",

    "did not explore",
    "does not explore",
    "not explored",

    "did not investigate",
    "does not investigate",
    "not investigated",

    "did not evaluate",
    "does not evaluate",
    "not evaluated",

    "unclear whether",
    "it is unclear whether",

    "has not been explored",
    "has not been investigated",
    "has not been evaluated",

    "remains unexplored",
    "remains uninvestigated",

    "no evidence that",
    "there is no evidence that",

    "the paper fails to",
    "the authors fail to",

    "the study fails to",
]


# Novelty claims are never valid research-gap evidence.

NOVELTY_PATTERNS = [

    "novelty",
    "unprecedented",
    "first of its kind",
    "first study",
    "first approach",
    "unexplored",
    "globally unexplored",
    "no previous work",
    "nobody has solved",
    "no one has solved",
    "never been solved",
]


# ============================================================
# VALIDATE LLM ANSWER
# ============================================================

def _no_gap_result() -> str:
    """
    The single, exact "nothing sufficiently supported" result.

    Used for every failure/empty path in validate_gap_answer so the
    caller always gets the same, predictable shape when there is no
    defensible research gap.
    """

    return json.dumps(
        {
            "limitations": [],
            "possible_gaps": [],
        },
        indent=2,
        ensure_ascii=False,
    )


def validate_gap_answer(
    answer: str,
    source_chunks: list[dict] | None = None,
    requested_paper_id: int | None = None,
) -> str:
    """
    Validate structured JSON returned by the LLM.

    The LLM output is NOT trusted blindly.

    Validation performs:

    1. JSON parsing.
    2. Root structure validation.
    3. Status validation.
    4. SOURCE validation - citation exists, is in range, and
       belongs to the requested paper_id (source_chunks /
       requested_paper_id).
    5. Groundedness validation - the statement must share actual
       vocabulary with the chunk text it cites, not just carry a
       well-formed but hallucinated citation.
    6. Limitation validation.
    7. Possible-gap validation.
    8. Limitation → gap source relationship validation.
    9. Solution-proposal rejection.
    10. Unsupported absence-of-evidence rejection.
    11. Novelty-claim rejection.
    """

    if not answer or not answer.strip():

        return _no_gap_result()

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

        data = json.loads(
            text
        )

    except json.JSONDecodeError:

        print(
            "\nINVALID GAP JSON FROM LLM"
        )

        print(
            "=" * 80
        )

        print(
            text
        )

        return _no_gap_result()

    # ---------------------------------------------------------
    # Root validation
    # ---------------------------------------------------------

    if not isinstance(
        data,
        dict,
    ):

        return _no_gap_result()

    limitations = data.get(
        "limitations",
        [],
    )

    possible_gaps = data.get(
        "possible_gaps",
        [],
    )

    if not isinstance(
        limitations,
        list,
    ):

        limitations = []

    if not isinstance(
        possible_gaps,
        list,
    ):

        possible_gaps = []

    # =========================================================
    # VALIDATE LIMITATIONS
    # =========================================================

    clean_limitations = []

    for item in limitations:

        if not isinstance(
            item,
            dict,
        ):
            continue

        statement = item.get(
            "statement"
        )

        status = item.get(
            "status"
        )

        source = item.get(
            "source"
        )

        if not statement:
            continue

        if not status:
            continue

        if not source:
            continue

        statement = str(
            statement
        ).strip()

        status = str(
            status
        ).strip().upper()

        normalized_source, cited_chunk = _resolve_source_chunk(
            source,
            source_chunks,
            requested_paper_id,
        )

        if not normalized_source:

            print(
                "\nREJECTED LIMITATION:"
                " invalid, out-of-range, or cross-paper"
                " SOURCE citation"
            )

            print(
                f"Source: {source}"
            )

            print(
                f"Statement: {statement}"
            )

            continue

        # -----------------------------------------------------
        # Validate status
        # -----------------------------------------------------

        if status not in VALID_STATUSES:
            continue

        # -----------------------------------------------------
        # Reject statements not grounded in the cited chunk
        # -----------------------------------------------------

        if not _statement_grounded_in_chunk(
            statement,
            cited_chunk.get("text", ""),
        ):

            print(
                "\nREJECTED LIMITATION:"
                " statement is not grounded in the cited"
                " source text"
            )

            print(
                f"Source: {normalized_source}"
            )

            print(
                f"Statement: {statement}"
            )

            continue

        # -----------------------------------------------------
        # Reject unsupported absence claims
        # -----------------------------------------------------

        lower_statement = statement.lower()

        if _contains_any(
            lower_statement,
            UNSUPPORTED_ABSENCE_PATTERNS,
        ):
            print(
                "\nREJECTED LIMITATION:"
                " unsupported absence-of-evidence claim"
            )

            print(
                statement
            )

            continue

        # -----------------------------------------------------
        # Reject novelty claims
        # -----------------------------------------------------

        if _contains_any(
            lower_statement,
            NOVELTY_PATTERNS,
        ):
            print(
                "\nREJECTED LIMITATION:"
                " novelty claim detected"
            )

            print(
                statement
            )

            continue

        clean_limitations.append(
            {
                "statement": statement,
                "status": status,
                "source": normalized_source,
            }
        )

    # =========================================================
    # BUILD ACCEPTED GAP SOURCES
    # =========================================================

    # A possible gap is allowed ONLY when the corresponding
    # limitation is:
    #
    #     PARTIALLY ADDRESSED
    #
    # or:
    #
    #     EXPLICITLY UNRESOLVED
    #
    # This is the key structural validation rule.

    allowed_gap_sources = {
        item["source"]
        for item in clean_limitations
        if item["status"]
        in {
            "PARTIALLY ADDRESSED",
            "EXPLICITLY UNRESOLVED",
        }
    }

    # A gap must describe the SAME underlying problem as the
    # limitation it comes from. This map lets us run a
    # conservative key-term overlap check per source below.

    limitation_statements_by_source: dict[str, list[str]] = {}

    for item in clean_limitations:

        limitation_statements_by_source.setdefault(
            item["source"],
            [],
        ).append(
            item["statement"]
        )

    # =========================================================
    # VALIDATE POSSIBLE GAPS
    # =========================================================

    clean_gaps = []

    for item in possible_gaps:

        if not isinstance(
            item,
            dict,
        ):
            continue

        statement = item.get(
            "statement"
        )

        source = item.get(
            "source"
        )

        if not statement:
            continue

        if not source:
            continue

        statement = str(
            statement
        ).strip()

        # A possible gap is a conservative RESTATEMENT of its
        # limitation in the limitation's own terminology (see the
        # gap system prompt) - it is expected to diverge from the
        # raw chunk wording, so we validate the citation itself
        # (exists, in range, correct paper) here, and check topical
        # relatedness against the LIMITATION statement it claims to
        # derive from (via _gap_shares_key_terms below), not against
        # the raw chunk text.
        normalized_source, _cited_chunk = _resolve_source_chunk(
            source,
            source_chunks,
            requested_paper_id,
        )

        if not normalized_source:

            print(
                "\nREJECTED GAP:"
                " invalid, out-of-range, or cross-paper"
                " SOURCE citation"
            )

            print(
                f"Source: {source}"
            )

            print(
                f"Statement: {statement}"
            )

            continue

        lower_statement = statement.lower()

        # -----------------------------------------------------
        # KEY RULE:
        #
        # Gap source MUST correspond to a limitation whose
        # status is PARTIALLY ADDRESSED or
        # EXPLICITLY UNRESOLVED.
        # -----------------------------------------------------

        if normalized_source not in allowed_gap_sources:

            print(
                "\nREJECTED GAP:"
            )

            print(
                "Source does not have an accepted "
                "unresolved/partially-addressed limitation."
            )

            print(
                f"Source: {normalized_source}"
            )

            print(
                f"Statement: {statement}"
            )

            continue

        # -----------------------------------------------------
        # Reject gaps that are clearly unrelated to the
        # limitation(s) reported for this source (conservative
        # key-term overlap - see _gap_shares_key_terms()).
        # -----------------------------------------------------

        if not _gap_shares_key_terms(
            statement,
            limitation_statements_by_source.get(
                normalized_source,
                [],
            ),
        ):

            print(
                "\nREJECTED GAP:"
            )

            print(
                "Statement shares no key terms with its "
                "source limitation and appears unrelated."
            )

            print(
                f"Source: {normalized_source}"
            )

            print(
                f"Statement: {statement}"
            )

            continue

        # -----------------------------------------------------
        # Reject solution proposals
        # -----------------------------------------------------

        if _contains_any(
            lower_statement,
            FORBIDDEN_SOLUTION_PATTERNS,
        ):

            print(
                "\nREJECTED GAP:"
            )

            print(
                "Solution/recommendation detected."
            )

            print(
                f"Source: {normalized_source}"
            )

            print(
                f"Statement: {statement}"
            )

            continue

        # -----------------------------------------------------
        # Reject action-oriented solution statements
        # -----------------------------------------------------

        if _looks_like_solution_statement(
            statement
        ):

            print(
                "\nREJECTED GAP:"
            )

            print(
                "Action-oriented solution statement detected."
            )

            print(
                f"Source: {normalized_source}"
            )

            print(
                f"Statement: {statement}"
            )

            continue

        # -----------------------------------------------------
        # Reject unsupported absence-of-evidence claims
        # -----------------------------------------------------

        if _contains_any(
            lower_statement,
            UNSUPPORTED_ABSENCE_PATTERNS,
        ):

            print(
                "\nREJECTED GAP:"
            )

            print(
                "Unsupported absence-of-evidence claim detected."
            )

            print(
                f"Source: {normalized_source}"
            )

            print(
                f"Statement: {statement}"
            )

            continue

        # -----------------------------------------------------
        # Reject novelty claims
        # -----------------------------------------------------

        if _contains_any(
            lower_statement,
            NOVELTY_PATTERNS,
        ):

            print(
                "\nREJECTED GAP:"
            )

            print(
                "Novelty claim detected."
            )

            print(
                f"Source: {normalized_source}"
            )

            print(
                f"Statement: {statement}"
            )

            continue

        # -----------------------------------------------------
        # Accept validated gap
        # -----------------------------------------------------

        clean_gaps.append(
            {
                "statement": statement,
                "source": normalized_source,
            }
        )

    # =========================================================
    # REMOVE DUPLICATE LIMITATIONS
    # =========================================================

    unique_limitations = []

    seen_limitations = set()

    for item in clean_limitations:

        key = (
            item["statement"].strip().lower(),
            item["status"],
            item["source"],
        )

        if key in seen_limitations:
            continue

        seen_limitations.add(
            key
        )

        unique_limitations.append(
            item
        )

    clean_limitations = unique_limitations

    # =========================================================
    # REMOVE DUPLICATE GAPS
    # =========================================================

    unique_gaps = []

    seen_gaps = set()

    for item in clean_gaps:

        key = (
            item["statement"].strip().lower(),
            item["source"],
        )

        if key in seen_gaps:
            continue

        seen_gaps.add(
            key
        )

        unique_gaps.append(
            item
        )

    clean_gaps = unique_gaps

    # =========================================================
    # FINAL SAFETY CHECK
    # =========================================================

    if not clean_limitations:

        return _no_gap_result()

    # ---------------------------------------------------------
    # Final defensive source/status check
    # ---------------------------------------------------------

    final_allowed_sources = {
        item["source"]
        for item in clean_limitations
        if item["status"]
        in {
            "PARTIALLY ADDRESSED",
            "EXPLICITLY UNRESOLVED",
        }
    }

    clean_gaps = [
        gap
        for gap in clean_gaps
        if gap["source"]
        in final_allowed_sources
    ]

    # =========================================================
    # RETURN CLEAN JSON
    # =========================================================
    #
    # If, after every validation pass, there are no accepted
    # limitations that actually qualify for a gap AND no gaps
    # survived, treat this the same as "nothing sufficiently
    # supported" rather than returning an empty-but-structurally
    # different payload.

    if not final_allowed_sources and not clean_gaps:

        return _no_gap_result()

    return json.dumps(
        {
            "limitations": clean_limitations,
            "possible_gaps": clean_gaps,
        },
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# DETECT RESEARCH GAPS
# ============================================================

async def detect_research_gaps(
    db: AsyncSession,
    paper_id: int | None = None,
    top_k: int = 5,
):
    """
    Detect conservative research gaps from paper evidence.

    Pipeline:

        Multiple targeted queries
                ↓
        Semantic + keyword retrieval
                ↓
        Deduplicate chunks
                ↓
        Select strongest evidence
                ↓
        Add neighboring chunks
                ↓
        Build evidence context
                ↓
        LLM evidence analysis
                ↓
        JSON validation
                ↓
        Possible research gaps
    """

    # ============================================================
    # 1. TARGETED GAP QUERIES
    # ============================================================

    gap_queries = [
        "limitations of the proposed method",
        "limitations of the model",
        "limitations of the experiments",
        "weaknesses of the proposed approach",
        "problems with the proposed approach",
        "performance degradation",
        "trade-offs and disadvantages",
        "future work",
        "future improvements",
        "unresolved problems",
        "challenges and difficulties",
        "what could be improved",
        "inability of the proposed method",
        "does not perform well",
        "remains difficult",
        "does not have a complete explanation",
    ]

    # ============================================================
    # 2. COLLECT UNIQUE RETRIEVED CHUNKS
    # ============================================================

    all_results = {}

    for query in gap_queries:

        print(
            f"\nGAP QUERY: {query}"
        )

        retrieved = await retrieve_chunks(
            db=db,
            query=query,
            paper_id=paper_id,
            top_k=top_k,
            min_similarity=0.20,
        )

        if retrieved is None:
            continue

        if not isinstance(
            retrieved,
            list,
        ):

            raise TypeError(
                "retrieve_chunks() must return "
                f"a list, got {type(retrieved)}"
            )

        for result in retrieved:

            if not isinstance(
                result,
                dict,
            ):
                continue

            if (
                "paper_id" not in result
                or "chunk_id" not in result
            ):
                continue

            key = (
                result["paper_id"],
                result["chunk_id"],
            )

            if key not in all_results:

                all_results[key] = result

            else:

                old_score = all_results[key].get(
                    "final_score",
                    all_results[key].get(
                        "similarity",
                        0.0,
                    )
                    or 0.0,
                )

                new_score = result.get(
                    "final_score",
                    result.get(
                        "similarity",
                        0.0,
                    )
                    or 0.0,
                )

                if new_score > old_score:

                    all_results[key] = result

    # ============================================================
    # 3. CONVERT DICT → LIST
    # ============================================================

    results = list(
        all_results.values()
    )

    # ------------------------------------------------------------
    # 3a. PAPER ISOLATION (defensive)
    #
    # retrieve_chunks() already filters by paper_id at the SQL
    # level, but we re-verify here so this pipeline can never
    # silently use evidence from another paper even if that
    # invariant is ever broken upstream.
    # ------------------------------------------------------------

    if paper_id is not None:

        before_count = len(results)

        results = [
            result
            for result in results
            if result.get("paper_id") == paper_id
        ]

        if len(results) != before_count:

            print(
                f"\nDISCARDED {before_count - len(results)} "
                f"CHUNK(S) BELONGING TO A DIFFERENT PAPER "
                f"(requested paper_id={paper_id})"
            )

    # ------------------------------------------------------------
    # 3b. REFERENCES / BIBLIOGRAPHY FILTER
    #
    # Never use bibliography, acknowledgments, or author-info
    # sections as gap evidence.
    # ------------------------------------------------------------

    before_count = len(results)

    results = [
        result
        for result in results
        if not _is_reference_chunk(result.get("text", ""))
        and not _is_visualization_chunk(result.get("text", ""))
    ]

    if len(results) != before_count:

        print(
            f"\nDISCARDED {before_count - len(results)} "
            "REFERENCE/BIBLIOGRAPHY/VISUALIZATION CHUNK(S)"
        )

    # ------------------------------------------------------------
    # 3c. MINIMUM FINAL-SCORE THRESHOLD
    #
    # A chunk may have cleared the per-query min_similarity bar
    # only because of the keyword bonus. Require a sensible
    # combined score before it is allowed to become gap evidence.
    # ------------------------------------------------------------

    before_count = len(results)

    results = [
        result
        for result in results
        if (result.get("final_score") or 0.0) >= MIN_GAP_FINAL_SCORE
    ]

    if len(results) != before_count:

        print(
            f"\nDISCARDED {before_count - len(results)} "
            f"CHUNK(S) BELOW MIN FINAL SCORE "
            f"({MIN_GAP_FINAL_SCORE})"
        )

    if not results:

        return {
            "paper_id": paper_id,
            "gaps": json.dumps(
                {
                    "limitations": [],
                    "possible_gaps": [],
                },
                indent=2,
                ensure_ascii=False,
            ),
            "sources": [],
        }

    # ============================================================
    # 4. SORT BY HYBRID SCORE (with limitation-evidence priority)
    # ============================================================

    results.sort(
        key=_candidate_sort_key,
        reverse=True,
    )

    # ============================================================
    # 5. SELECT STRONGEST CANDIDATES
    # ============================================================

    candidate_results = results[:5]

    print(
        "\nSELECTED GAP CANDIDATES"
    )

    print(
        "=" * 80
    )

    for candidate in candidate_results:

        score = candidate.get(
            "final_score",
            0.0,
        )

        if score is None:
            score = 0.0

        print(
            f"Paper ID: {candidate['paper_id']} | "
            f"Chunk ID: {candidate['chunk_id']} | "
            f"Score: {float(score):.4f}"
        )

    # ============================================================
    # 6. ADD NEIGHBORING CHUNKS
    # ============================================================

    expanded_results = await expand_with_neighbors(
        db=db,
        retrieved_chunks=candidate_results,
        window=1,
    )

    if not isinstance(
        expanded_results,
        list,
    ):

        raise TypeError(
            "expand_with_neighbors() must return "
            f"a list, got {type(expanded_results)}"
        )

    # ============================================================
    # 7. REMOVE DUPLICATES AGAIN
    # ============================================================

    unique_expanded = {}

    for chunk in expanded_results:

        if not isinstance(
            chunk,
            dict,
        ):
            continue

        if (
            "paper_id" not in chunk
            or "chunk_id" not in chunk
        ):
            continue

        key = (
            chunk["paper_id"],
            chunk["chunk_id"],
        )

        if key not in unique_expanded:

            unique_expanded[key] = chunk

    expanded_results = list(
        unique_expanded.values()
    )

    # ------------------------------------------------------------
    # 7a. PAPER ISOLATION (defensive, post-neighbor-expansion)
    # ------------------------------------------------------------

    if paper_id is not None:

        before_count = len(expanded_results)

        expanded_results = [
            chunk
            for chunk in expanded_results
            if chunk.get("paper_id") == paper_id
        ]

        if len(expanded_results) != before_count:

            print(
                f"\nDISCARDED {before_count - len(expanded_results)} "
                f"NEIGHBOR CHUNK(S) BELONGING TO A DIFFERENT PAPER "
                f"(requested paper_id={paper_id})"
            )

    # ------------------------------------------------------------
    # 7b. REFERENCES / BIBLIOGRAPHY FILTER (post-neighbor-expansion)
    # ------------------------------------------------------------

    before_count = len(expanded_results)

    expanded_results = [
        chunk
        for chunk in expanded_results
        if not _is_reference_chunk(chunk.get("text", ""))
        and not _is_visualization_chunk(chunk.get("text", ""))
    ]

    if len(expanded_results) != before_count:

        print(
            f"\nDISCARDED {before_count - len(expanded_results)} "
            "REFERENCE/BIBLIOGRAPHY/VISUALIZATION NEIGHBOR CHUNK(S)"
        )

    if not expanded_results:

        return {
            "paper_id": paper_id,
            "gaps": json.dumps(
                {
                    "limitations": [],
                    "possible_gaps": [],
                },
                indent=2,
                ensure_ascii=False,
            ),
            "sources": [],
        }

    # ============================================================
    # 8. SORT CONTEXT
    # ============================================================

    expanded_results.sort(
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

    expanded_results = expanded_results[:15]

    print(
        "\nFINAL GAP CONTEXT CHUNKS"
    )

    print(
        "=" * 80
    )

    for chunk in expanded_results:

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
        expanded_results,
        max_chunks=15,
    )

    if not context.strip():

        return {
            "paper_id": paper_id,
            "gaps": json.dumps(
                {
                    "limitations": [],
                    "possible_gaps": [],
                },
                indent=2,
                ensure_ascii=False,
            ),
            "sources": expanded_results,
        }

    # ============================================================
    # 10. BUILD LLM PROMPT
    # ============================================================

    prompt = build_gap_prompt(
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
        raw_answer = (
            '{"limitations":[],'
            '"possible_gaps":[]}'
        )

    print(
        "\nRAW GAP LLM OUTPUT"
    )

    print(
        "=" * 80
    )

    print(
        raw_answer
    )

    # ============================================================
    # 12. VALIDATE LLM ANSWER
    # ============================================================

    answer = validate_gap_answer(
        raw_answer,
        source_chunks=expanded_results,
        requested_paper_id=paper_id,
    )

    # ============================================================
    # 12.5 VERIFY EACH LIMITATION AGAINST ITS SOURCE CHUNK
    # ============================================================

    answer = await _verify_limitations(
        answer,
        source_chunks=expanded_results,
        requested_paper_id=paper_id,
    )

    # ============================================================
    # 13. FINAL RESULT
    # ============================================================

    return {
        "paper_id": paper_id,
        "gaps": answer,
        "sources": expanded_results,
    }