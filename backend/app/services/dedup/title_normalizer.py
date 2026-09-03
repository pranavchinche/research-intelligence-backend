#D:\FYP\main\backend\app\services\dedup\paper_deduplicator.py

import re # math function for regularize expression


def normalize_title(title: str) -> str:
    title = title.lower()

    title = re.sub(
        r"[^a-z0-9\s]",
        "",
        title
    )

    title = re.sub(
        r"\s+",
        " ",
        title
    )

    return title.strip()
