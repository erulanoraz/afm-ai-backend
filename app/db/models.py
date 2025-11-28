from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Integer, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid


class Base(DeclarativeBase):
    pass


# ---------- Таблица файлов ----------
class File(Base):
    __tablename__ = "files"

    file_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False, index=True)
    case_id = Column(String, index=True, nullable=True)
    s3_key = Column(String, nullable=False)  # 🔥 УБРАЛ unique=True
    received_at = Column(DateTime(timezone=True), server_default=func.now())
    ocr_confidence = Column(Float, nullable=True)
    chunks_count = Column(Integer, default=0, nullable=True)  # ✅ добавь это поле

    chunks = relationship("Chunk", back_populates="file", cascade="all, delete-orphan")



# ---------- Таблица чанков ----------
class Chunk(Base):
    __tablename__ = "chunks"

    chunk_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id = Column(UUID(as_uuid=True), ForeignKey("files.file_id", ondelete="CASCADE"), index=True)
    page = Column(Integer, nullable=True)
    start_offset = Column(Integer, nullable=True)
    end_offset = Column(Integer, nullable=True)
    text = Column(Text, nullable=False)
    meta = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    evidence = Column(JSON, nullable=True)
    entities = Column(JSON, nullable=True)
    events = Column(JSON, nullable=True)
    facts = Column(JSON, nullable=True)
    SLG = Column(JSON, nullable=True)
    section_detection = Column(JSON, nullable=True)

    # обратная связь с File
    file = relationship("File", back_populates="chunks")
