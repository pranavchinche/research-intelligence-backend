from sqlalchemy import Column, Integer, Text, ForeignKey, String
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.models.base import Base


class PaperChunk(Base):
    __tablename__ = "paper_chunks"

    id = Column(Integer, primary_key=True, index=True)

    paper_id = Column(
        Integer,
        ForeignKey("papers.id"),
        nullable=False
    )

    chunk_id = Column(
        Integer,
        nullable=False
    )

    page_number = Column(
        Integer,
        nullable=True
    )

    text = Column(
        Text,
        nullable=False
    )

    model = Column(
        String(100),
        nullable=False
    )

    embedding = Column(
        Vector(384),
        nullable=True
    )

    paper = relationship(
        "Paper",
        back_populates="chunks"
    )