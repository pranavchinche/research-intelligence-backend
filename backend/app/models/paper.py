#D:\FYP\main\backend\app\models\paper.py
from sqlalchemy import Column, Integer, String, Text, DateTime
from app.models.base import Base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

class Paper(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)

    title = Column(String(500), nullable=False)

    abstract = Column(Text, nullable=True)
    authors = Column(Text, nullable=True)

    doi = Column(String(255), unique=True, nullable=True)

    source = Column(String(50), nullable=False)
    source_id = Column(String(255), nullable=True)

    arxiv_id = Column(String(255), unique=True, nullable=True)

    published_date = Column(DateTime, nullable=True)
    updated_date = Column(DateTime, nullable=True)

    categories = Column(Text, nullable=True)

    pdf_url = Column(String(1000), nullable=True)
    pdf_path = Column(String(1000), nullable=True)
    drive_file_id = Column(String(255), nullable=True)

    full_text = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    chunks = relationship(
        "PaperChunk",
        back_populates="paper",
        cascade="all, delete-orphan"
    )
    user_links = relationship(
        "UserPaper",
        back_populates="paper",
        cascade="all, delete-orphan"
    )