# app/api/v1/upload.py
"""
Evidence Engine INGEST v3.1 — Upload API
ИСПРАВЛЕНИЯ:
✅ Удалена зависимость от app.tasks.ingest (дублирующего модуля)
✅ Используется app.services.ingest_service напрямую
✅ Правильная интеграция с Vector Store (через vector_tasks)
"""

import uuid
import os
import tempfile
import zipfile
import shutil
import logging
import re
from typing import List, Optional, Dict

from fastapi import (
    APIRouter,
    Depends,
    UploadFile,
    File as FastAPIFile,
    HTTPException,
)
from sqlalchemy.orm import Session

from app.db import get_db
from app.db.models import File
from app.services.parser import extract_text_from_file
from app.utils.config import settings

# ✅ ИСПРАВЛЕНИЕ: Используем ingest_service напрямую вместо Celery task
from app.services.ingest_service import process_any_file

# ============================================================
# Константы
# ============================================================
MAX_FILE_SIZE_MB = getattr(settings, "MAX_FILE_SIZE_MB", 100)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

DEFAULT_INGEST_DIR = os.path.join(tempfile.gettempdir(), "afm_ingest")
INGEST_DIR = getattr(settings, "INGEST_DIR", DEFAULT_INGEST_DIR)
os.makedirs(INGEST_DIR, exist_ok=True)

router = APIRouter(prefix="/upload", tags=["Upload"])
logger = logging.getLogger(__name__)

CASE_ID_REGEX = r"(\d{15})"  # номер ЕРДР / дела — 15 цифр подряд


# ============================================================
# Вспомогательные функции
# ============================================================

def _extract_case_id_from_name(name: str) -> Optional[str]:
    """
    Извлекает номер ЕРДР / дела из имени файла.
    Ищем 15 подряд идущих цифр.
    """
    if not name:
        return None
    m = re.search(CASE_ID_REGEX, name)
    return m.group(1) if m else None


def _extract_case_id_from_text(text: str) -> Optional[str]:
    """
    Извлекает номер ЕРДР / дела из текста документа.
    """
    if not text:
        return None
    m = re.search(CASE_ID_REGEX, text)
    return m.group(1) if m else None


def _detect_case_id_from_pdf(pdf_path: str, filename: str) -> Optional[str]:
    """
    Извлекает case_id из текста PDF (через extract_text_from_file).
    Используется для PDF внутри ZIP и одиночных PDF.
    """
    try:
        text = extract_text_from_file(pdf_path) or ""
    except Exception as e:
        logger.warning(f"⚠️ Не удалось извлечь текст из PDF {filename}: {e}")
        return None

    if not text.strip():
        return None

    case_id = _extract_case_id_from_text(text)
    if case_id:
        logger.info(f"🔎 case_id={case_id} найден внутри PDF: {filename}")
    return case_id


def _detect_case_id_for_file(
    file_path: str,
    filename: str,
    outer_case_id: Optional[str] = None,
) -> Optional[str]:
    """
    Evidence Engine style detector:
    1) Извлекает case_id из имени файла
    2) Для PDF/DOCX/TXT — вытаскивает текст и ищет case_id
    3) Fallback — используется outer_case_id (для файлов в ZIP)
    """
    if not filename:
        filename = ""

    ext = os.path.splitext(filename)[1].lower()

    # 1) По имени файла
    case_id = _extract_case_id_from_name(filename)
    if case_id:
        logger.info(f"🔎 case_id={case_id} найден в названии: {filename}")
        return case_id

    # 2) По тексту файла — PDF/DOCX/TXT
    text: str = ""

    try:
        if ext in [".pdf", ".docx", ".txt"]:
            # extract_text_from_file умеет работать и с PDF, и с DOCX/TXT
            text = extract_text_from_file(file_path) or ""
    except Exception as e:
        logger.warning(f"⚠️ Не удалось извлечь текст из {filename}: {e}")
        text = ""

    if text.strip():
        case_id_from_text = _extract_case_id_from_text(text)
        if case_id_from_text:
            logger.info(
                f"🔎 case_id={case_id_from_text} найден в тексте: {filename}"
            )
            return case_id_from_text

    # 3) Fallback — outer_case_id (для файлов в ZIP)
    if outer_case_id:
        logger.info(
            f"ℹ️ Для {filename} используем outer_case_id={outer_case_id}"
        )
        return outer_case_id

    logger.info(f"⚠️ case_id не найден: {filename}")
    return None


def _validate_file_size(file: UploadFile) -> None:
    """Проверка размера файла."""
    if hasattr(file, "size") and file.size:
        if file.size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Файл {file.filename} слишком большой ({file.size / 1024 / 1024:.1f} МБ). "
                    f"Максимум: {MAX_FILE_SIZE_MB} МБ"
                ),
            )


# ============================================================
# Main Upload Endpoint
# ============================================================

@router.post("/")
async def upload_files(
    files: List[UploadFile] = FastAPIFile(...),
    db: Session = Depends(get_db),
):
    """
    Evidence Engine INGEST v3.1:
    
    Процесс:
    1. Загружаем файл (максимально быстро)
    2. Создаём запись File в БД и коммитим
    3. Вызываем process_any_file() синхронно (OCR + Chunker)
    4. process_any_file() создаёт чанки в БД
    5. Вызываем enqueue_chunk_vectorization() для каждого чанка
    6. Celery vector_tasks worker индексирует в Weaviate
    
    Результат: Полная pipeline от upload до Vector Store
    """
    results: List[dict] = []
    case_ids_map: Dict[str, List[str]] = {}

    logger.info(f"📤 Загружается файлов: {len(files)}")

    for file in files:
        temp_path: Optional[str] = None

        try:
            # ========== 1. Проверка размера ==========
            try:
                _validate_file_size(file)
            except HTTPException as e:
                results.append({
                    "file_id": None,
                    "filename": file.filename,
                    "chunks_created": 0,
                    "error": e.detail,
                    "status": "failed",
                })
                continue

            # ========== 2. Временное сохранение ==========
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f"_{file.filename}"
            ) as tmp:
                temp_path = tmp.name
                content = await file.read()
                if len(content) > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Файл слишком большой. Максимум: {MAX_FILE_SIZE_MB} МБ"
                    )
                tmp.write(content)

            ext = os.path.splitext(file.filename)[1].lower()
            logger.info(f"📥 Загружен {file.filename}, размер {len(content)} байт")

            # ========== 3. ZIP – распаковка и обработка каждого файла ==========
            if ext == ".zip":
                extract_dir = tempfile.mkdtemp(prefix="unzipped_")
                try:
                    with zipfile.ZipFile(temp_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                except zipfile.BadZipFile:
                    raise HTTPException(status_code=400, detail="ZIP повреждён")

                outer_case_id = _extract_case_id_from_name(file.filename)
                logger.info(f"📦 ZIP распакован, найдено файлов...")

                zip_inner_ids: List[str] = []

                for root, _, inner_files in os.walk(extract_dir):
                    for inner_name in inner_files:
                        inner_path = os.path.join(root, inner_name)
                        inner_ext = os.path.splitext(inner_name)[1].lower()

                        if inner_ext not in [".pdf", ".docx", ".txt"]:
                            continue

                        inner_file_id = uuid.uuid4()

                        try:
                            # Определяем case_id
                            case_id_to_save = _detect_case_id_for_file(
                                file_path=inner_path,
                                filename=inner_name,
                                outer_case_id=outer_case_id,
                            )

                            # Создаём запись File в БД
                            new_file = File(
                                file_id=inner_file_id,
                                filename=inner_name,
                                case_id=case_id_to_save,
                                s3_key=f"s3://afm-originals/{inner_name}",
                                ocr_confidence=0.0,
                                chunks_count=0,
                            )
                            db.add(new_file)
                            db.commit()
                            db.refresh(new_file)
                            logger.info(f"  ✅ File запись создана: {inner_file_id}")

                            # 🔥 КРИТИЧЕСКОЕ: Синхронно обрабатываем файл (OCR + Chunker)
                            try:
                                chunks_created = process_any_file(
                                    file_path=inner_path,
                                    file_id=inner_file_id,
                                    db=db
                                )
                                logger.info(f"  ✅ Обработано: {chunks_created} чанков")
                                
                                # Обновляем запись File с количеством чанков
                                new_file.chunks_count = chunks_created
                                db.commit()
                                
                            except Exception as ocr_err:
                                logger.error(f"  ❌ Ошибка обработки файла: {ocr_err}")
                                chunks_created = 0

                            if case_id_to_save:
                                case_ids_map.setdefault(case_id_to_save, []).append(
                                    str(inner_file_id)
                                )

                            results.append({
                                "file_id": str(inner_file_id),
                                "filename": inner_name,
                                "chunks_created": chunks_created,
                                "case_id": case_id_to_save,
                                "status": "completed" if chunks_created > 0 else "warning",
                            })
                            zip_inner_ids.append(str(inner_file_id))

                        except Exception as e:
                            db.rollback()
                            logger.error(f"❌ Ошибка {inner_name}: {e}")
                            results.append({
                                "file_id": str(inner_file_id),
                                "filename": inner_name,
                                "chunks_created": 0,
                                "error": str(e),
                                "status": "failed",
                            })

                # Очищаем временные директории
                shutil.rmtree(extract_dir, ignore_errors=True)
                os.remove(temp_path)
                temp_path = None
                
                results.append({
                    "file_id": None,
                    "filename": file.filename,
                    "type": "zip_summary",
                    "files_processed": len(zip_inner_ids),
                    "status": "completed",
                })
                continue

            # ========== 4. PDF – основной формат ==========
            if ext == ".pdf":
                file_id = uuid.uuid4()
                try:
                    case_id_extracted = _detect_case_id_for_file(
                        file_path=temp_path,
                        filename=file.filename,
                        outer_case_id=None,
                    )

                    # Создаём запись File в БД
                    new_file = File(
                        file_id=file_id,
                        filename=file.filename,
                        case_id=case_id_extracted,
                        s3_key=f"s3://afm-originals/{file.filename}",
                        ocr_confidence=0.0,
                        chunks_count=0,
                    )
                    db.add(new_file)
                    db.commit()
                    db.refresh(new_file)
                    logger.info(f"✅ File запись создана: {file_id}")

                    # 🔥 КРИТИЧЕСКОЕ: Синхронно обрабатываем (OCR + Chunker)
                    chunks_created = 0
                    try:
                        chunks_created = process_any_file(
                            file_path=temp_path,
                            file_id=file_id,
                            db=db
                        )
                        logger.info(f"✅ Обработано: {chunks_created} чанков")
                        
                        # Обновляем запись
                        new_file.chunks_count = chunks_created
                        db.commit()
                        
                    except Exception as ocr_err:
                        logger.error(f"❌ Ошибка OCR: {ocr_err}")
                        chunks_created = 0

                    if case_id_extracted:
                        case_ids_map.setdefault(case_id_extracted, []).append(
                            str(file_id)
                        )

                    results.append({
                        "file_id": str(file_id),
                        "filename": file.filename,
                        "chunks_created": chunks_created,
                        "case_id": case_id_extracted,
                        "status": "completed" if chunks_created > 0 else "warning",
                    })

                except Exception as e:
                    db.rollback()
                    logger.error(f"❌ Ошибка PDF {file.filename}: {e}")
                    results.append({
                        "file_id": str(file_id),
                        "filename": file.filename,
                        "chunks_created": 0,
                        "error": str(e),
                        "status": "failed",
                    })
                continue

            # ========== 5. DOCX / TXT ==========
            if ext in [".docx", ".txt"]:
                file_id = uuid.uuid4()
                try:
                    case_id_extracted = _detect_case_id_for_file(
                        file_path=temp_path,
                        filename=file.filename,
                        outer_case_id=None,
                    )

                    # Создаём запись File в БД
                    new_file = File(
                        file_id=file_id,
                        filename=file.filename,
                        case_id=case_id_extracted,
                        s3_key=f"s3://afm-originals/{file.filename}",
                        ocr_confidence=1.0,  # ← DOCX/TXT уже текст, OCR не нужен
                        chunks_count=0,
                    )
                    db.add(new_file)
                    db.commit()
                    db.refresh(new_file)
                    logger.info(f"✅ File запись создана: {file_id}")

                    # 🔥 КРИТИЧЕСКОЕ: Синхронно обрабатываем
                    chunks_created = 0
                    try:
                        chunks_created = process_any_file(
                            file_path=temp_path,
                            file_id=file_id,
                            db=db
                        )
                        logger.info(f"✅ Обработано: {chunks_created} чанков")
                        
                        # Обновляем запись
                        new_file.chunks_count = chunks_created
                        db.commit()
                        
                    except Exception as ocr_err:
                        logger.error(f"❌ Ошибка обработки: {ocr_err}")
                        chunks_created = 0

                    if case_id_extracted:
                        case_ids_map.setdefault(case_id_extracted, []).append(
                            str(file_id)
                        )

                    results.append({
                        "file_id": str(file_id),
                        "filename": file.filename,
                        "chunks_created": chunks_created,
                        "case_id": case_id_extracted,
                        "status": "completed" if chunks_created > 0 else "warning",
                    })

                except Exception as e:
                    db.rollback()
                    logger.error(f"❌ Ошибка {file.filename}: {e}")
                    results.append({
                        "file_id": str(file_id),
                        "filename": file.filename,
                        "chunks_created": 0,
                        "error": str(e),
                        "status": "failed",
                    })
                continue

            # ========== 6. Неподдерживаемый формат ==========
            results.append({
                "file_id": None,
                "filename": file.filename,
                "chunks_created": 0,
                "error": f"Неподдерживаемый формат: {ext}",
                "status": "failed",
            })

        except Exception as e:
            logger.error(f"❌ Критическая ошибка: {e}", exc_info=True)
            results.append({
                "file_id": None,
                "filename": file.filename,
                "error": str(e),
                "chunks_created": 0,
                "status": "failed",
            })

        finally:
            # Очищаем временный файл
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    # ========== Финальный ответ ==========
    successful_files = sum(1 for r in results if r["status"] == "completed")
    warning_files = sum(1 for r in results if r["status"] == "warning")
    failed_files = sum(1 for r in results if r["status"] == "failed")

    return {
        "uploaded_files": len([r for r in results if r.get("file_id")]),
        "successful": successful_files,
        "warnings": warning_files,
        "failed": failed_files,
        "results": results,
        "case_ids": case_ids_map if case_ids_map else None,
    }
