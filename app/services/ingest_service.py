# =====================================================================
# INGEST SERVICE 7.0
# ChatGPT-Style Evidence Pipeline
# PostgreSQL-only ingest + Celery async vectorization
# =====================================================================

import os
import uuid
import magic
import logging
from datetime import datetime
from sqlalchemy.orm import Session

from app.db.models import File, Chunk
from app.storage.s3_client import upload_to_s3

from app.services.ocr_worker import extract_text_from_pdf, run_tesseract_ocr
from app.services.ocr_corrector import correct_ocr_text
from app.services.parser import extract_text_from_file
from app.services.chunker import process_text_into_chunks
from app.utils.config import settings

# Celery async indexing
from app.tasks.vector_tasks import enqueue_chunk_vectorization

logger = logging.getLogger("INGEST7")


# =====================================================================
# NORMALIZATION
# =====================================================================

def normalize_text(text: str) -> str:
    import re
    if not text:
        return ""

    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# =====================================================================
# ChatGPT Evidence Pipeline — text cleaning
# =====================================================================

def unified_pipeline(raw_text: str) -> str:
    """
    Evidence Engine style:
    1) normalize
    2) LLM OCR correction (STRICT)
    """
    if not raw_text or not raw_text.strip():
        return ""

    normalized = normalize_text(raw_text)
    corrected = correct_ocr_text(normalized)

    return corrected.strip()


# =====================================================================
# PROCESS ANY FILE
# =====================================================================

def process_any_file(file_path: str, file_id, db: Session) -> int:
    """
    Unified pipeline:
    PDF → OCR → LLM Corrector → Chunker
    DOCX/TXT → extract → LLM Corrector → Chunker

    NOTE:
    - ZERO Weaviate calls here
    - chunker saves ONLY to PostgreSQL
    - We then schedule Celery tasks for vector indexing
    """

    ext = os.path.splitext(file_path)[1].lower()

    # ---------------------------- PDF ----------------------------
    if ext == ".pdf":
        logger.info("📄 PDF detected → OCR → LLM Correct → Chunker")

        raw_text = extract_text_from_pdf(file_path)

        if not raw_text or len(raw_text.strip()) < 50:
            logger.warning("⚠ PDF layer weak → fallback Tesseract full OCR")
            raw_text = run_tesseract_ocr(file_path, page_num=None)

        cleaned = unified_pipeline(raw_text)

        return process_text_into_chunks(file_id=file_id, text=cleaned, db=db)

    # ------------------------- DOCX/TXT ---------------------------
    elif ext in [".docx", ".txt"]:
        logger.info("📝 DOCX/TXT → extract → LLM Correct → Chunker")

        raw_text = extract_text_from_file(file_path) or ""

        if not raw_text.strip():
            raw_text = extract_text_from_pdf(file_path)

        cleaned = unified_pipeline(raw_text)

        return process_text_into_chunks(file_id=file_id, text=cleaned, db=db)

    # ------------------------- UNSUPPORTED -------------------------
    else:
        logger.error(f"❌ Unsupported file type: {ext}")
        return 0


# =====================================================================
# INGEST DOCUMENT (main entrypoint)
# =====================================================================

def ingest_document(file_bytes: bytes, filename: str, db: Session, uploader=None, case_id=None):
    """
    UPLOAD → S3 → DB(File) → TEMP FILE → INGEST (OCR → CORRECT → CHUNKER) → PostgreSQL
    AND THEN:
    create Celery tasks for vector indexing in Weaviate
    """

    logger.info(f"📥 INGEST START: {filename}")

    # MIME TYPE
    content_type = magic.Magic(mime=True).from_buffer(file_bytes)

    # Upload to S3
    s3_key = upload_to_s3(file_bytes, filename)

    # Create File record
    file_id = uuid.uuid4()

    file_obj = File(
        file_id=file_id,
        filename=filename,
        s3_key=s3_key,
        case_id=case_id,
        uploader=uploader,
        received_at=datetime.utcnow(),
        metadata={"content_type": content_type},
    )
    db.add(file_obj)
    db.commit()
    db.refresh(file_obj)

    # Save temp file
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    temp_path = os.path.join(settings.TEMP_DIR, f"{file_id}_{filename}")

    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    # Run ingest
    try:
        chunks_created = process_any_file(
            file_path=temp_path,
            file_id=file_id,
            db=db
        )

        # -----------------------------
        # AFTER INGEST → CREATE TASKS
        # -----------------------------
        logger.info(f"🔄 Creating Celery vector tasks for chunks (file_id={file_id})")

        chunk_records = (
            db.query(Chunk.chunk_id)
            .filter(Chunk.file_id == file_id)
            .all()
        )

        for (chunk_id,) in chunk_records:
            enqueue_chunk_vectorization.delay(str(chunk_id))

        logger.info(f"🚀 Celery tasks created: {len(chunk_records)}")

    except Exception as e:
        logger.error(f"❌ INGEST ERROR: {e}", exc_info=True)
        raise

    finally:
        try:
            os.remove(temp_path)
        except:
            pass

    return {
        "file_id": str(file_id),
        "filename": filename,
        "case_id": case_id,
        "s3_key": s3_key,
        "chunks_created": chunks_created,
        "content_type": content_type,
        "pipeline": "ChatGPT Evidence Pipeline v1.0",
        "message": "INGEST completed. Vector indexing scheduled via Celery."
    }
