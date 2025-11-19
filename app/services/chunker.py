# app/services/chunker.py
import os
import re
import logging
import uuid
from typing import Optional
from PyPDF2 import PdfReader, PdfWriter
from sqlalchemy.orm import Session

from app.db.models import Chunk
from app.services.ocr_worker import (
    extract_text_from_pdf,
    run_tesseract_ocr,
    run_tesseract_ocr_image,
)
from app.services.parser import extract_text_from_file

from pdf2image import convert_from_path
from app.utils.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# 🧩 Безопасное приведение к UUID
# ============================================================

def ensure_uuid(value) -> Optional[uuid.UUID]:
    try:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except Exception:
        logger.error(f"❌ Некорректный UUID: {value}")
        return None


# ============================================================
# 📄 OCR постраничный (исправленный)
# ============================================================

def process_pdf_page_ocr(file_path: str, page_num: int) -> str:
    """
    Извлекает ОДНУ страницу PDF → JPEG → OCR (PIL Image).
    Это самый корректный метод.
    """
    try:
        pages = convert_from_path(
            file_path,
            dpi=300,
            poppler_path=settings.POPPLER_PATH,
            first_page=page_num,
            last_page=page_num,
            fmt="jpeg",
        )

        if not pages:
            logger.warning(f"⚠️ convert_from_path не вернул стр. {page_num}")
            return ""

        image = pages[0]
        text = run_tesseract_ocr_image(image, page_num=page_num, use_preprocessing=True)
        return text or ""

    except Exception as e:
        logger.error(f"❌ Ошибка OCR convert_from_path page={page_num}: {e}")
        return ""


# ============================================================
# 📄 OCR обработка PDF (обычная)
# ============================================================

def process_pdf_with_ocr(file_path: str, file_id, db: Session) -> int:
    file_id = ensure_uuid(file_id)
    if not file_id:
        return 0

    if not os.path.exists(file_path):
        logger.error(f"❌ Файл не найден: {file_path}")
        return 0

    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        logger.info(f"📖 Открыт PDF: {os.path.basename(file_path)}, страниц={total_pages}")
    except Exception as e:
        logger.error(f"❌ Ошибка открытия PDF {file_path}: {e}")

        full_text = extract_text_from_pdf(file_path, dpi=300, use_preprocessing=True)
        if full_text.strip():
            return process_text_into_chunks(file_id, full_text, db)
        return 0

    chunks_created = 0

    for page_num in range(1, total_pages + 1):
        text = process_pdf_page_ocr(file_path, page_num)

        if not text.strip():
            logger.warning(f"⚠️ OCR пустой на стр. {page_num}")
            continue

        chunk = Chunk(
            chunk_id=uuid.uuid4(),
            file_id=file_id,
            page=page_num,
            start_offset=0,
            end_offset=len(text),
            text=text.strip(),
        )
        db.add(chunk)
        chunks_created += 1

    db.flush()
    logger.info(f"📄 process_pdf_with_ocr: создано {chunks_created} чанков")
    return chunks_created


# ============================================================
# 🧠 SMART OCR (исправленный)
# ============================================================

def process_pdf_with_smart_ocr(file_path: str, file_id, db: Session) -> int:
    file_id = ensure_uuid(file_id)
    if not file_id:
        return 0

    if not os.path.exists(file_path):
        logger.error(f"❌ Файл не найден: {file_path}")
        return 0

    chunks_created = 0
    reader = None

    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        ...

        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
                if not text or len(text.strip()) < 30:
                    logger.info(f"[SMART OCR] Страница {i}: текста мало — запускаем Tesseract OCR")
                    text = run_tesseract_ocr(
                        file_path=file_path,
                        page_num=i,
                        use_preprocessing=True,
                    ) or ""
                if not text.strip():
                    logger.warning(f"⚠️ OCR не смог извлечь текст на стр. {i}")
                    continue

                chunk = Chunk(
                    chunk_id=uuid.uuid4(),
                    file_id=file_id,
                    page=i,
                    start_offset=0,
                    end_offset=len(text),
                    text=text.strip(),
                )
                db.add(chunk)
                chunks_created += 1

            except Exception as e:
                logger.error(f"❌ Ошибка обработки страницы {i}: {e}", exc_info=True)
                continue

        # 🟩 FAIL-SAFE: если после всех страниц нет ни одного чанка → fallback
        if chunks_created == 0:
            logger.warning("⚠️ SMART OCR не дал чанков. Запускаю fallback OCR…")
            try:
                fallback_text = extract_text_from_pdf(
                    file_path, dpi=300, use_preprocessing=True
                )
            except Exception as fe:
                logger.error(f"❌ Fallback OCR тоже не сработал: {fe}")
                fallback_text = ""

            if not fallback_text or not fallback_text.strip():
                fallback_text = " "  # минимальный placeholder

            chunk = Chunk(
                chunk_id=uuid.uuid4(),
                file_id=file_id,
                page=1,
                start_offset=0,
                end_offset=len(fallback_text),
                text=fallback_text.strip(),
            )
            db.add(chunk)
            chunks_created = 1
            logger.info("🟧 Создан fallback/placeholder-чанк")

        db.flush()
        logger.info(f"✅ SMART OCR завершён — создано {chunks_created} чанков")
        return chunks_created

    except Exception as e:
        logger.error(f"❌ Критическая ошибка SMART OCR: {e}", exc_info=True)
        # 🔴 ВАЖНО: НИКАКОГО db.rollback() ЗДЕСЬ
        return 0

    finally:
        try:
            if reader and hasattr(reader, "stream") and reader.stream:
                reader.stream.close()
        except Exception:
            pass



# ============================================================
# 📑 Разделение текста на чанки
# ============================================================

def process_text_into_chunks(file_id, text: str, db: Session, min_len=50, page_start=1) -> int:
    if not text or not text.strip():
        return 0

    file_id = ensure_uuid(file_id)
    if not file_id:
        return 0

    paragraphs = []
    if "\n\n" in text:
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > min_len]

    if not paragraphs:
        paragraphs = [
            p.strip()
            for p in re.split(r"(?<=[.!?])\s+(?=[А-ЯA-Z])", text)
            if len(p.strip()) > min_len
        ]

    if not paragraphs:
        return 0

    chunks_created = 0

    for idx, chunk_text in enumerate(paragraphs, start=page_start):
        chunk = Chunk(
            chunk_id=uuid.uuid4(),
            file_id=file_id,
            page=idx,
            start_offset=0,
            end_offset=len(chunk_text),
            text=chunk_text,
        )
        db.add(chunk)
        chunks_created += 1

    db.flush()
    return chunks_created


# ============================================================
# 📦 Универсальный обработчик
# ============================================================

def process_any_file(file_path: str, file_id, db: Session) -> int:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        chunks_created = process_pdf_with_smart_ocr(file_path, file_id, db)
        if chunks_created == 0:
            chunks_created = process_pdf_with_ocr(file_path, file_id, db)
        return chunks_created

    elif ext in [".docx", ".txt"]:
        text = extract_text_from_file(file_path)
        if not text.strip():
            text = extract_text_from_pdf(file_path, dpi=300)
        return process_text_into_chunks(file_id, text, db)

    else:
        logger.warning(f"⛔ Неподдерживаемый тип файла: {ext}")
        return 0
