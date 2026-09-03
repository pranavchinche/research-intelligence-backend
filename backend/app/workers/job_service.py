"""Job management service.

Provides functions to create, update, and query jobs in the
database. Used by the ingestion pipeline and analysis features.
"""

from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.job import Job
from app.workers.retry_policy import (
    can_retry,
    calculate_next_retry,
    get_retry_info,
)


async def create_job(
    db: AsyncSession,
    paper_id: int,
    job_type: str,
    max_retries: int = 3,
) -> Job:
    """Create a new job in the queued state."""
    job = Job(
        paper_id=paper_id,
        job_type=job_type,
        status="queued",
        progress=0,
        max_retries=max_retries,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def start_job(
    db: AsyncSession,
    job: Job,
) -> None:
    """Mark a job as running."""
    job.status = "running"
    job.progress = 0
    await db.commit()


async def update_progress(
    db: AsyncSession,
    job: Job,
    progress: int,
    message: str = "",
) -> None:
    """Update job progress (0-100)."""
    job.progress = min(100, max(0, progress))
    job.progress_message = message
    await db.commit()


async def complete_job(
    db: AsyncSession,
    job: Job,
    provider: str = "",
) -> None:
    """Mark a job as succeeded."""
    job.status = "succeeded"
    job.progress = 100
    job.completed_at = datetime.now(timezone.utc)
    if provider:
        job.llm_provider_used = provider
    await db.commit()


async def fail_job(
    db: AsyncSession,
    job: Job,
    error: str,
) -> None:
    """Mark a job as failed. Schedules retry if applicable."""
    job.error = error[:2000]

    if can_retry(job.retry_count, job.max_retries, error):
        job.status = "retrying"
        job.retry_count += 1
        job.next_retry_at = calculate_next_retry(job.retry_count)
    else:
        job.status = "failed"
        job.completed_at = datetime.now(timezone.utc)

    await db.commit()


async def get_job(
    db: AsyncSession,
    job_id: int,
) -> Job | None:
    """Fetch a job by ID."""
    result = await db.execute(
        select(Job).where(Job.id == job_id)
    )
    return result.scalar_one_or_none()


async def get_jobs_for_paper(
    db: AsyncSession,
    paper_id: int,
) -> list[Job]:
    """Fetch all jobs for a given paper."""
    result = await db.execute(
        select(Job)
        .where(Job.paper_id == paper_id)
        .order_by(Job.created_at.desc())
    )
    return list(result.scalars().all())


async def get_pending_retries(
    db: AsyncSession,
) -> list[Job]:
    """Fetch jobs that are due for retry."""
    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(Job)
        .where(
            Job.status == "retrying",
            Job.next_retry_at <= now,
        )
        .order_by(Job.next_retry_at)
    )
    return list(result.scalars().all())
