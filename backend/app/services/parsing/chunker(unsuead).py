#D:\FYP\main\backend\app\services\parsing\chunker.py
def chunk_text(
    pages: list[dict],
    chunk_size: int = 700,
    overlap: int = 100,
) -> list[dict]:

    chunks = []

    for page in pages:
        page_number = page["page_number"]
        text = page["text"].strip()

        if not text:
            continue

        words = text.split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_words = words[start:end]

            chunk = " ".join(chunk_words).strip()

            if chunk:
                chunks.append(
                    {
                        "chunk_id": len(chunks) + 1,
                        "text": chunk,
                        "page_number": page_number,
                    }
                )

            if end >= len(words):
                break

            start = end - overlap

    return chunks