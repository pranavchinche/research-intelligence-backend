"""Job model for async processing pipeline.

Per the architecture (Section 21), every long-running operation
runs as a row in the jobs table with status, progress, and retry
tracking.
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
)
from sqlalchemy.sql import func
from app.models.base import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)

    paper_id = Column(
        Integer,
        ForeignKey("papers.id"),
        nullable=True,
    )

    job_type = Column(
        String(50),
        nullable=False,
    )

    status = Column(
        String(30),
        nullable=False,
        default="queued",
    )

    progress = Column(
        Integer,
        nullable=False,
        default=0,
    )

    progress_message = Column(
        Text,
        nullable=True,
    )

    retry_count = Column(
        Integer,
        nullable=False,
        default=0,
    )

    max_retries = Column(
        Integer,
        nullable=False,
        default=3,
    )

    next_retry_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    llm_provider_used = Column(
        String(50),
        nullable=True,
    )

    error = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )
