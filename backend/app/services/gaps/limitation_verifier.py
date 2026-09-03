from app.services.llm import get_llm_answer


LIMITATION_VERIFIER_PROMPT = """
You are a strict research-paper evidence verifier.

Your ONLY task is to determine whether the supplied research-paper
passage explicitly states a limitation, weakness, difficulty,
challenge, trade-off, degradation, unresolved issue, or stated need
for improvement.

Do NOT infer a limitation.

A passage is VALID only if the paper explicitly indicates that
something is problematic, difficult, weak, degraded, unresolved,
limited, or needs improvement.

A normal technical description is INVALID.

A positive result is INVALID.

A method description is INVALID.

A statement that something "could be useful" is INVALID unless the
paper explicitly describes the current situation as a limitation.

Return EXACTLY this format:

VALID: YES
EVIDENCE: <exact sentence from the supplied passage>

OR

VALID: NO
EVIDENCE: NONE

Rules:
- Use ONLY the supplied passage.
- Never use outside knowledge.
- Never rewrite the evidence.
- Never introduce a solution.
- Never claim novelty.
- Never claim that a problem is unexplored.
"""


def build_verification_prompt(text: str) -> str:

    return f"""
{LIMITATION_VERIFIER_PROMPT}

RESEARCH-PAPER PASSAGE
======================

{text}

VERIFY THE PASSAGE.
"""


async def verify_limitation(text: str) -> dict:

    prompt = build_verification_prompt(text)

    try:
        answer = await get_llm_answer(prompt)
    except RuntimeError:
        return {
            "valid": False,
            "evidence": None,
            "raw_answer": "",
        }

    answer = answer.strip()

    if answer.upper().startswith("VALID: YES"):
        valid = True
    else:
        valid = False

    evidence = None

    for line in answer.splitlines():

        if line.upper().startswith("EVIDENCE:"):

            evidence = line.split(
                ":", 
                1
            )[1].strip()

            if evidence.upper() == "NONE":
                evidence = None

            break

    return {
        "valid": valid,
        "evidence": evidence,
        "raw_answer": answer,
    }