from app.db.models import File, Chunk
from app.services.ocr_worker import ocr_pdf_bytes, ocr_image_bytes
from app.services.chunker import sliding_window_chunks
from app.storage.s3_client import upload_to_s3
from app.utils.config import settings
from sqlalchemy.orm import Session
import magic, uuid
from datetime import datetime

def normalize_text(text: str) -> str:
    import re
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def ingest_document(file_bytes: bytes, filename: str, db: Session, uploader=None, case_id=None):
    """OCR → нормализация → чанкинг → запись в S3 + БД"""
    content_type = magic.Magic(mime=True).from_buffer(file_bytes)

    s3_key = upload_to_s3(file_bytes, filename)
    file_obj = File(
        file_id=uuid.uuid4(),
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

    text, conf = "", None
    if filename.lower().endswith(".pdf"):
        text, conf, _ = ocr_pdf_bytes(file_bytes)
    elif content_type.startswith("image/"):
        text, conf = ocr_image_bytes(file_bytes)
    else:
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            pass

    text = normalize_text(text)
    if not text:
        raise ValueError("Не удалось извлечь текст")

    file_obj.ocr_confidence = conf
    db.add(file_obj)
    db.commit()

    chunks = sliding_window_chunks(text, target_tokens=settings.CHUNK_TOKENS, overlap_tokens=settings.CHUNK_OVERLAP)
    for start_tok, end_tok, chunk_text in chunks:
        ch = Chunk(file_id=file_obj.file_id, start_offset=start_tok, end_offset=end_tok, text=chunk_text)
        db.add(ch)
    db.commit()

    return {
        "file_id": str(file_obj.file_id),
        "chunks": len(chunks),
        "s3_key": s3_key,
        "content_type": content_type,
    }
