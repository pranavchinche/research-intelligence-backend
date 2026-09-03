from typing import List, Dict
import re


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def clean_text(text: str) -> str:
    """
    Clean common PDF extraction artifacts while preserving
    meaningful research-paper text.
    """

    if not text:
        return ""

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove common PDF tokenizer artifacts
    text = re.sub(r"<EOS>", " ", text)
    text = re.sub(r"<pad>", " ", text)

    # Remove repeated spaces created after removing artifacts
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """
    Basic sentence splitting suitable for research-paper text.
    """

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Create semantically cleaner chunks.

    The chunker prefers sentence boundaries instead of blindly
    cutting every 1000 characters.
    """

    if not text or not text.strip():
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    text = clean_text(text)

    if not text:
        return []

    sentences = split_into_sentences(text)

    if not sentences:
        return []

    chunks = []
    current_sentences = []
    current_length = 0

    for sentence in sentences:

        sentence_length = len(sentence)

        # If adding the sentence would exceed the chunk size,
        # finalize the current chunk.
        if (
            current_sentences
            and current_length + sentence_length + 1 > chunk_size
        ):
            chunk = " ".join(current_sentences).strip()

            if chunk:
                chunks.append(chunk)

            # Keep a small overlap from the end of the previous chunk.
            overlap_sentences = []
            overlap_length = 0

            for previous in reversed(current_sentences):

                if overlap_length + len(previous) > chunk_overlap:
                    break

                overlap_sentences.insert(0, previous)
                overlap_length += len(previous) + 1

            current_sentences = overlap_sentences
            current_length = overlap_length

        current_sentences.append(sentence)
        current_length += sentence_length + 1

    # Add final chunk
    if current_sentences:

        chunk = " ".join(current_sentences).strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def is_low_quality_chunk(text: str) -> bool:
    """
    Detect chunks that are unlikely to contain useful
    research-paper prose.

    This does NOT try to understand the research content.
    It only removes obvious extraction noise.
    """

    if not text:
        return True

    cleaned = text.strip()

    if len(cleaned) < 80:
        return True

    words = cleaned.split()

    if len(words) < 15:
        return True

    # Excessive PDF/tokenizer artifacts
    artifact_count = (
        cleaned.count("<EOS>")
        + cleaned.count("<pad>")
    )

    if artifact_count > 2:
        return True

    # Very high ratio of isolated tokens often indicates
    # extracted figure/table text.
    isolated_tokens = sum(
        1
        for word in words
        if len(word) <= 2
    )

    if len(words) >= 20:
        isolated_ratio = isolated_tokens / len(words)

        if isolated_ratio > 0.45:
            return True

    # Bibliography/reference-like chunks
    lower_text = cleaned.lower()

    reference_indicators = [
        "proceedings of",
        "computational linguistics",
        "journal of",
        "vol.",
        "pages ",
        "acl,",
        "arxiv preprint",
    ]

    reference_matches = sum(
        indicator in lower_text
        for indicator in reference_indicators
    )

    if reference_matches >= 2:
        return True

    return False


def chunk_pages(
    pages: List[Dict],
) -> List[Dict]:
    """
    Convert page-level PDF text into semantic chunks.

    Each chunk retains its source page number.
    Obvious PDF extraction noise is filtered out.
    """

    results = []

    chunk_id = 1

    for page in pages:

        page_number = page["page_number"]
        text = page.get("text", "")

        if not text or not text.strip():
            continue

        page_chunks = split_text(text)

        for chunk_text in page_chunks:

            if is_low_quality_chunk(chunk_text):
                continue

            results.append(
                {
                    "chunk_id": chunk_id,
                    "page_number": page_number,
                    "text": chunk_text,
                }
            )

            chunk_id += 1

    return results