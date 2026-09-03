"""
Backfill author metadata for existing papers whose PDFs were uploaded
before author extraction was fixed.

Usage:
    python scripts/backfill_authors.py --dry-run   # preview only, no DB writes
    python scripts/backfill_authors.py              # execute updates

Idempotent: running twice will find zero eligible papers after the first run.
"""

import argparse
import asyncio
import os
import sys
import textwrap

from sqlalchemy import or_, select, func

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.database import AsyncSessionlocal
from app.models.paper import Paper
from app.services.parsing.pdf_parser import extract_pdf_text


async def backfill(dry_run: bool = False) -> None:
    async with AsyncSessionlocal() as session:
        result = await session.execute(
            select(Paper).where(
                or_(
                    Paper.authors.is_(None),
                    func.trim(Paper.authors) == "",
                )
            )
        )
        papers = list(result.scalars().all())

        total = len(papers)
        updated = 0
        skipped_pdf_missing = 0
        unresolved = 0

        updated_details: list[tuple[int, str, str]] = []
        unresolved_details: list[tuple[int, str, str]] = []

        for paper in papers:
            pdf_path = paper.pdf_path
            title = paper.title or "(no title)"

            if not pdf_path or not os.path.isfile(pdf_path):
                skipped_pdf_missing += 1
                unresolved_details.append(
                    (paper.id, title, "PDF not found on disk")
                )
                continue

            try:
                parsed = extract_pdf_text(pdf_path)
            except Exception as exc:
                skipped_pdf_missing += 1
                unresolved_details.append(
                    (paper.id, title, f"PDF parse error: {exc}")
                )
                continue

            raw_author = (parsed.get("metadata") or {}).get("author")
            author = raw_author.strip() if raw_author else ""

            if not author:
                unresolved += 1
                unresolved_details.append(
                    (paper.id, title, "No author found in PDF")
                )
                continue

            updated += 1
            updated_details.append((paper.id, title, author))

            if not dry_run:
                paper.authors = author

        if not dry_run and updated > 0:
            await session.commit()

    _print_report(
        dry_run=dry_run,
        total=total,
        updated=updated,
        skipped_pdf_missing=skipped_pdf_missing,
        unresolved=unresolved,
        updated_details=updated_details,
        unresolved_details=unresolved_details,
    )


def _print_report(
    dry_run: bool,
    total: int,
    updated: int,
    skipped_pdf_missing: int,
    unresolved: int,
    updated_details: list[tuple[int, str, str]],
    unresolved_details: list[tuple[int, str, str]],
) -> None:
    mode = "DRY RUN" if dry_run else "LIVE"
    sep = "=" * 56

    print(f"\n{sep}")
    print(f"  Author Backfill Report  [{mode}]")
    print(sep)
    print(f"  Papers checked:          {total}")
    print(f"  Successfully updated:    {updated}")
    print(f"  Skipped (PDF missing):   {skipped_pdf_missing}")
    print(f"  No author found:         {unresolved}")
    print(sep)

    if updated_details:
        print("\n--- Updated Papers ---")
        for pid, title, author in updated_details:
            wrapped_title = textwrap.fill(title, width=70, subsequent_indent="  ")
            print(f"  ID: {pid}")
            print(f"  Title: {wrapped_title}")
            print(f"  Author: {author}")
            print()

    if unresolved_details:
        print("--- Unresolved Papers ---")
        for pid, title, reason in unresolved_details:
            wrapped_title = textwrap.fill(title, width=70, subsequent_indent="  ")
            print(f"  ID: {pid}")
            print(f"  Title: {wrapped_title}")
            print(f"  Reason: {reason}")
            print()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Backfill missing author metadata from stored PDFs."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to the database.",
    )
    args = parser.parse_args()
    asyncio.run(backfill(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
