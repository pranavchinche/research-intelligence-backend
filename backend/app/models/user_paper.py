# D:\FYP\main\backend\app\models\user_paper.py

from sqlalchemy import (
    Column,
    Integer,
    DateTime,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.models.base import Base


class UserPaper(Base):
    __tablename__ = "user_papers"

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "paper_id",
            name="uq_user_papers_user_id_paper_id",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    paper_id = Column(
        Integer,
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    user = relationship(
        "User",
        back_populates="papers",
    )

    paper = relationship(
        "Paper",
        back_populates="user_links",
    )
