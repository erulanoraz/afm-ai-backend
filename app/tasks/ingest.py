# app/tasks/ingest.py

import logging
import os
import uuid

from celery import shared_task
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.db.models import File
from app.services.chunker import process_any_file, process_text_into_chunks
from app.services.parser import extract_text_from_file

logger = logging.getLogger("INGEST_TASKS")


@shared_task(name="ingest.process_file")
def process_file_task(file_id_str: str, path: str, ext: str) -> None:
    """
    Фоновая обработка файла под Chunker 7.0 (PostgreSQL-only):

    • PDF → SMART OCR 7.0 → token-based chunker → evidence JSON
    • DOCX/TXT → parser → chunker → evidence JSON
    • Чанки пишутся ТОЛЬКО в PostgreSQL
    • Векторизация уходит в Celery очередь "vectors"
    """
    session: Session = SessionLocal()

    try:
        file_id = uuid.UUID(file_id_str)

        file_obj: File | None = (
            session.query(File).filter(File.file_id == file_id).one_or_none()
        )
        if not file_obj:
            logger.error(f"❌ File с id={file_id} не найден в БД")
            return

        ext = ext.lower()
        logger.info(
            f"▶️ [TASK] Старт обработки: {file_obj.filename} "
            f"(file_id={file_id}, ext={ext}, path={path})"
        )

        # === PDF ======================================================
        if ext == ".pdf":
            chunks_created = process_any_file(
                file_path=path,
                file_id=file_id,
                db=session,
            )

        # === DOCX / TXT ==============================================
        elif ext in [".docx", ".txt"]:
            text = extract_text_from_file(path) or ""
            if text.strip():
                chunks_created = process_text_into_chunks(
                    file_id=file_id,
                    text=text,
                    db=session,
                )
            else:
                logger.warning(
                    f"⚠️ Пустой текст у {file_obj.filename}, пропуск."
                )
                chunks_created = 0

        # === Unsupported =============================================
        else:
            logger.warning(
                f"⚠️ Неподдерживаемый формат {ext} в Celery-таске"
            )
            return

        # Обновляем запись
        file_obj.chunks_count = chunks_created
        file_obj.ocr_confidence = file_obj.ocr_confidence or 0.9
        session.commit()

        logger.info(
            f"✅ [TASK] Обработка завершена: {file_obj.filename}, "
            f"file_id={file_id}, chunks={chunks_created}"
        )

    except Exception as e:
        session.rollback()
        logger.exception(
            f"❌ Ошибка в Celery-таске process_file_task "
            f"(file_id={file_id_str}): {e}"
        )
        raise

    finally:
        session.close()

        # Удаляем временный файл
        try:
            if path and os.path.exists(path):
                os.remove(path)
                logger.info(f"🧹 Удалён временный файл: {path}")
        except Exception:
            pass
