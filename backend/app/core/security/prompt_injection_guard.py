"""Prompt injection defense for retrieved paper text.

Per the architecture (Section 24), retrieved chunk text is treated
as data, never as instructions. This module flags text resembling
instruction-override patterns before it reaches the LLM.
"""

import re
import logging


logger = logging.getLogger(__name__)


# Patterns that suggest prompt injection attempts
_INJECTION_PATTERNS = [
    re.compile(
        r"ignore\s+(all\s+)?(previous|prior|above)\s+"
        r"(instructions?|prompts?|rules?|guidelines?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"you\s+are\s+now\s+(a|an|the)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"disregard\s+(all\s+)?(previous|prior|above)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"forget\s+(all\s+)?(previous|prior|your)\s+"
        r"(instructions?|prompts?|rules?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"new\s+instructions?:",
        re.IGNORECASE,
    ),
    re.compile(
        r"system\s*(prompt|message)\s*:",
        re.IGNORECASE,
    ),
    re.compile(
        r"act\s+as\s+if\s+you\s+(have|had)\s+no\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"override\s+(your|the)\s+(previous|prior|existing)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"do\s+not\s+follow\s+(your|the)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"<\|im_start\|>|<\|im_end\|>",
        re.IGNORECASE,
    ),
]

# Maximum number of flaggable patterns in a single chunk
_MAX_FLAGS = 3


def scan_for_injection(text: str) -> dict:
    """Scan text for prompt injection patterns.

    Returns:
        {
            "flagged": bool,
            "flags": list[str],  # matched pattern descriptions
            "cleaned_text": str,  # text with flags noted
        }
    """
    if not text:
        return {
            "flagged": False,
            "flags": [],
            "cleaned_text": text,
        }

    flags = []

    for i, pattern in enumerate(_INJECTION_PATTERNS):
        if pattern.search(text):
            flags.append(f"injection_pattern_{i}")

    flagged = len(flags) > 0

    if flagged:
        logger.warning(
            "Prompt injection patterns detected in text "
            "(%d flags): %s",
            len(flags),
            flags[:_MAX_FLAGS],
        )

    return {
        "flagged": flagged,
        "flags": flags,
        "cleaned_text": text,
    }


def guard_retrieved_chunks(
    chunks: list[dict],
) -> list[dict]:
    """Apply injection defense to a list of retrieved chunks.

    Each chunk is scanned; flagged chunks are still included
    (not silently stripped) but marked with an injection flag
    so callers can display a warning.
    """
    guarded = []
    for chunk in chunks:
        text = chunk.get("text", "")
        scan = scan_for_injection(text)
        guarded_chunk = dict(chunk)
        guarded_chunk["injection_flagged"] = scan["flagged"]
        guarded_chunk["injection_flags"] = scan["flags"]
        guarded.append(guarded_chunk)
    return guarded
