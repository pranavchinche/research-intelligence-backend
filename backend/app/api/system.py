"""System and operational endpoints.

Per the architecture (Section 6), these are unauthenticated
read-only status endpoints useful for the keep-alive cron
and for showing an honest "degraded mode" banner in the UI.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.config import settings
from app.services.connectivity.health_checker import health_checker
from app.services.connectivity.connectivity_guard import (
    connectivity_guard,
)
from app.services.llm.provider_registry import provider_registry
from app.core.security.secrets import validate_required_secrets
from app.workers.job_service import get_job


router = APIRouter(tags=["System"])


# ------------------------------------------------------------------
# GET /health — dependency-aware health check
# ------------------------------------------------------------------

@router.get("/health")
async def health():
    """Overall health check with per-dependency status."""
    statuses = await health_checker.check_all()
    warnings = validate_required_secrets()

    all_ok = all(s["reachable"] for s in statuses.values())

    return {
        "status": "healthy" if all_ok else "degraded",
        "services": statuses,
        "warnings": warnings,
    }


# ------------------------------------------------------------------
# GET /system/mode — online / degraded / offline
# ------------------------------------------------------------------

@router.get("/system/mode")
async def system_mode():
    """Current system operating mode."""
    mode = await connectivity_guard.evaluate(force=False)
    detail = connectivity_guard.get_status_detail()
    return detail


# ------------------------------------------------------------------
# GET /system/llm-provider — active provider + fallback chain
# ------------------------------------------------------------------

@router.get("/system/llm-provider")
async def llm_provider_status():
    """Which LLM provider is currently active."""
    chain = provider_registry.chain
    active = provider_registry.get_active_provider_name()
    providers = provider_registry.get_provider_status()

    return {
        "active_provider": active,
        "fallback_chain": chain,
        "providers": providers,
        "last_used": provider_registry.last_used_provider,
    }


# ------------------------------------------------------------------
# GET /jobs/{job_id} — job status polling
# ------------------------------------------------------------------

@router.get("/jobs/{job_id}")
async def job_status(
    job_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Poll job status, progress, and retry info."""
    job = await get_job(db, job_id)

    if job is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    return {
        "id": job.id,
        "paper_id": job.paper_id,
        "job_type": job.job_type,
        "status": job.status,
        "progress": job.progress,
        "progress_message": job.progress_message,
        "retry_count": job.retry_count,
        "max_retries": job.max_retries,
        "llm_provider_used": job.llm_provider_used,
        "error": job.error,
        "created_at": str(job.created_at) if job.created_at else None,
        "completed_at": str(job.completed_at) if job.completed_at else None,
    }
