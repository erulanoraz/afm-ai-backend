# app/api/v1/upload.py
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

# Celery-таск фоновой обработки файла (OCR + Chunker)
from app.tasks.ingest import process_file_task

# ============================================================
# Константы
# ============================================================
MAX_FILE_SIZE_MB = getattr(settings, "MAX_FILE_SIZE_MB", 100)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Директория, где будут лежать файлы для фоновой обработки
DEFAULT_INGEST_DIR = os.path.join(tempfile.gettempdir(), "afm_ingest")
INGEST_DIR = getattr(settings, "INGEST_DIR", DEFAULT_INGEST_DIR)
os.makedirs(INGEST_DIR, exist_ok=True)

router = APIRouter(prefix="/upload", tags=["Upload"])
logger = logging.getLogger(__name__)


# ============================================================
# Вспомогательные функции
# ============================================================

def _extract_case_id_from_name(name: str) -> Optional[str]:
    """
    Пытается извлечь номер ЕРДР / дела из ИМЕНИ файла / архива.
    Ищем 15 подряд идущих цифр.
    """
    if not name:
        return None
    m = re.search(r"(\d{15})", name)
    return m.group(1) if m else None


def _extract_case_id_from_text(text: str) -> Optional[str]:
    """
    Пытается извлечь номер ЕРДР / дела из ТЕКСТА документа.
    """
    if not text:
        return None
    m = re.search(r"(\d{15})", text)
    return m.group(1) if m else None


def _detect_case_id_for_file(
    file_path: str,
    filename: str,
    outer_case_id: Optional[str] = None,
) -> Optional[str]:
    """
    Evidence Engine style detector:

    1) пробуем вытащить case_id из имени файла;
    2) если DOCX/TXT — пробуем вытащить текст и найти там;
    3) если не нашли — используем outer_case_id (для файлов внутри ZIP).
    """
    if not filename:
        filename = ""

    ext = os.path.splitext(filename)[1].lower()

    # 1) По имени файла
    case_id = _extract_case_id_from_name(filename)
    if case_id:
        logger.info(f"🔎 case_id={case_id} найден в названии файла: {filename}")
        return case_id

    text: str = ""

    # 2) По тексту файла — ТОЛЬКО для DOCX/TXT
    if ext in [".docx", ".txt"]:
        try:
            text = extract_text_from_file(file_path) or ""
        except Exception as e:
            logger.warning(
                f"⚠️ Не удалось извлечь текст из файла {filename} для поиска case_id: {e}"
            )
            text = ""

    if text.strip():
        case_id_from_text = _extract_case_id_from_text(text)
        if case_id_from_text:
            logger.info(
                f"🔎 case_id={case_id_from_text} найден в тексте файла: {filename}"
            )
            return case_id_from_text

    # 3) Fallback — берем case_id снаружи (например, из имени ZIP)
    if outer_case_id:
        logger.info(
            f"ℹ️ Для файла {filename} используем outer_case_id={outer_case_id} "
            f"(по имени/тексту не найдено)"
        )
        return outer_case_id

    logger.info(f"⚠️ case_id не найден ни в имени, ни в тексте файла: {filename}")
    return None


def _validate_file_size(file: UploadFile) -> None:
    if hasattr(file, "size") and file.size:
        if file.size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Файл {file.filename} слишком большой ({file.size / 1024 / 1024:.1f} МБ). "
                    f"Максимальный размер: {MAX_FILE_SIZE_MB} МБ"
                ),
            )


def _store_for_ingest(src_path: str, file_id: uuid.UUID, ext: str) -> str:
    """
    Перекладываем файл в постоянную ingest-директорию,
    чтобы фоновый Celery-таск мог с ним работать уже после ответа API.
    """
    os.makedirs(INGEST_DIR, exist_ok=True)
    dst_path = os.path.join(INGEST_DIR, f"{file_id}{ext}")
    # если файл уже есть — перезапишем
    if os.path.exists(dst_path):
        os.remove(dst_path)
    shutil.move(src_path, dst_path)
    return dst_path


def _enqueue_ingest_job(file_id: uuid.UUID, stored_path: str, ext: str) -> None:
    """
    Кидаем задачу в Celery: обработать файл (OCR/Chunker и т.д.).
    ВАЖНО: на этом этапе запись File уже должна быть закоммичена в БД.
    """
    try:
        process_file_task.delay(str(file_id), stored_path, ext)
        logger.info(
            f"📨 Celery ingest task поставлена в очередь: file_id={file_id}, path={stored_path}"
        )
    except Exception as e:
        # Даже если Celery не запущен, загрузка файлов не должна падать.
        logger.error(f"❌ Не удалось поставить ingest-таск в очередь для {file_id}: {e}")


# ============================================================
# Evidence Engine INGEST v3.0
# ============================================================

@router.post("/")
async def upload_files(
    files: List[UploadFile] = FastAPIFile(...),
    db: Session = Depends(get_db),
):
    """
    Evidence Engine INGEST v3.0:

    • Загрузка файлов максимально быстрая: API только пишет записи File в БД
      и перекладывает файлы в ingest-директорию.
    • Тяжёлый OCR/Chunker выполняются в Celery (фоново).
    • Для КАЖДОГО файла:
        - File создаётся и фиксируется (db.commit) до запуска Celery.
        - Celery всегда видит запись в БД (нет ошибки "File ... не найден").
    """
    results: List[dict] = []
    case_ids_map: Dict[str, List[str]] = {}

    logger.info(f"📤 Загружается файлов: {len(files)}")

    for file in files:
        temp_path: Optional[str] = None

        try:
            # 1) Проверка размера
            try:
                _validate_file_size(file)
            except HTTPException as e:
                results.append(
                    {
                        "file_id": None,
                        "filename": file.filename,
                        "chunks_created": 0,
                        "error": e.detail,
                        "status": "failed",
                    }
                )
                continue

            # 2) Временное сохранение файла
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f"_{file.filename}"
            ) as tmp:
                temp_path = tmp.name
                content = await file.read()
                if len(content) > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Файл {file.filename} слишком большой. "
                            f"Максимум: {MAX_FILE_SIZE_MB} МБ"
                        ),
                    )
                tmp.write(content)

            ext = os.path.splitext(file.filename)[1].lower()
            logger.info(f"📥 Загружен {file.filename}, размер {len(content)} байт")

            # ============================================================
            # 3) ZIP – распаковываем, каждый inner-файл регистрируем в БД
            #      и отправляем в Celery отдельно
            # ============================================================
            if ext == ".zip":
                extract_dir = tempfile.mkdtemp(prefix="unzipped_")
                try:
                    with zipfile.ZipFile(temp_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                except zipfile.BadZipFile:
                    raise HTTPException(status_code=400, detail="ZIP файл повреждён")

                outer_case_id = _extract_case_id_from_name(file.filename)
                if outer_case_id:
                    logger.info(
                        f"🔎 outer_case_id={outer_case_id} найден в имени ZIP {file.filename}"
                    )
                else:
                    logger.info(f"ℹ️ В имени ZIP {file.filename} номер дела не найден")

                zip_inner_ids: List[str] = []

                for root, _, inner_files in os.walk(extract_dir):
                    for inner_name in inner_files:
                        inner_path = os.path.join(root, inner_name)
                        inner_ext = os.path.splitext(inner_name)[1].lower()

                        if inner_ext not in [".pdf", ".docx", ".txt"]:
                            continue

                        inner_file_id = uuid.uuid4()

                        try:
                            # 3.1 Определяем case_id
                            case_id_to_save = _detect_case_id_for_file(
                                file_path=inner_path,
                                filename=inner_name,
                                outer_case_id=outer_case_id,
                            )

                            # 3.2 Создаём запись в БД и СРАЗУ коммитим
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

                            # 3.3 Перекладываем файл в ingest-директорию
                            stored_path = _store_for_ingest(
                                inner_path, inner_file_id, inner_ext
                            )

                            # 3.4 Кидаем ingest-задачу в Celery
                            _enqueue_ingest_job(
                                inner_file_id, stored_path, inner_ext
                            )

                            if case_id_to_save:
                                case_ids_map.setdefault(case_id_to_save, []).append(
                                    str(inner_file_id)
                                )

                            results.append(
                                {
                                    "file_id": str(inner_file_id),
                                    "filename": inner_name,
                                    "chunks_created": 0,
                                    "case_id": case_id_to_save,
                                    "s3_key": f"s3://afm-originals/{inner_name}",
                                    "status": "queued",
                                }
                            )
                            zip_inner_ids.append(str(inner_file_id))

                        except Exception as e:
                            db.rollback()
                            logger.error(f"❌ Ошибка файла в ZIP {inner_name}: {e}")
                            results.append(
                                {
                                    "file_id": str(inner_file_id),
                                    "filename": inner_name,
                                    "chunks_created": 0,
                                    "error": str(e),
                                    "status": "failed",
                                }
                            )

                # ZIP как "обёртку" тоже отражаем в ответе (summary)
                results.append(
                    {
                        "file_id": None,
                        "filename": file.filename,
                        "type": "zip_summary",
                        "files_processed": len(zip_inner_ids),
                        "chunks_created": 0,
                        "case_id": outer_case_id,
                        "status": "queued",
                    }
                )

                # исходники уже перенесены в INGEST_DIR, эту папку можно удалить
                shutil.rmtree(extract_dir, ignore_errors=True)
                # temp_path тоже больше не нужен
                os.remove(temp_path)
                temp_path = None
                continue

            # ============================================================
            # 4) PDF – создаём File, коммитим, отправляем в Celery
            # ============================================================
            if ext == ".pdf":
                file_id = uuid.uuid4()
                try:
                    case_id_extracted = _detect_case_id_for_file(
                        file_path=temp_path,
                        filename=file.filename,
                        outer_case_id=None,
                    )

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

                    # Перекладываем PDF в ingest-директорию
                    stored_path = _store_for_ingest(temp_path, file_id, ext)
                    temp_path = None

                    # Кидаем фоновую обработку
                    _enqueue_ingest_job(file_id, stored_path, ext)

                    if case_id_extracted:
                        case_ids_map.setdefault(case_id_extracted, []).append(
                            str(file_id)
                        )

                    results.append(
                        {
                            "file_id": str(file_id),
                            "filename": file.filename,
                            "chunks_created": 0,
                            "case_id": case_id_extracted,
                            "s3_key": f"s3://afm-originals/{file.filename}",
                            "status": "queued",
                        }
                    )

                except Exception as e:
                    db.rollback()
                    logger.error(f"❌ Ошибка PDF {file.filename}: {e}")
                    results.append(
                        {
                            "file_id": str(file_id),
                            "filename": file.filename,
                            "chunks_created": 0,
                            "error": str(e),
                            "status": "failed",
                        }
                    )
                continue

            # ============================================================
            # 5) DOCX / TXT – аналогично PDF, но без OCR
            # ============================================================
            if ext in [".docx", ".txt"]:
                file_id = uuid.uuid4()
                try:
                    case_id_extracted = _detect_case_id_for_file(
                        file_path=temp_path,
                        filename=file.filename,
                        outer_case_id=None,
                    )

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

                    # Перекладываем файл в ingest-директорию
                    stored_path = _store_for_ingest(temp_path, file_id, ext)
                    temp_path = None

                    # Ставим фоновую задачу
                    _enqueue_ingest_job(file_id, stored_path, ext)

                    if case_id_extracted:
                        case_ids_map.setdefault(case_id_extracted, []).append(
                            str(file_id)
                        )

                    results.append(
                        {
                            "file_id": str(file_id),
                            "filename": file.filename,
                            "chunks_created": 0,
                            "case_id": case_id_extracted,
                            "status": "queued",
                        }
                    )

                except Exception as e:
                    db.rollback()
                    logger.error(f"❌ Ошибка обработки {file.filename}: {e}")
                    results.append(
                        {
                            "file_id": str(file_id),
                            "filename": file.filename,
                            "chunks_created": 0,
                            "error": str(e),
                            "status": "failed",
                        }
                    )
                continue

            # ============================================================
            # 6) Неподдерживаемый формат
            # ============================================================
            results.append(
                {
                    "file_id": None,
                    "filename": file.filename,
                    "chunks_created": 0,
                    "error": f"Неподдерживаемый формат: {ext}",
                    "status": "failed",
                }
            )

        except Exception as e:
            logger.error(f"❌ Критическая ошибка: {e}", exc_info=True)
            results.append(
                {
                    "file_id": None,
                    "filename": file.filename,
                    "error": str(e),
                    "chunks_created": 0,
                    "status": "failed",
                }
            )

        finally:
            # temp_path удаляем только если он ещё не был передан в ingest-директорию
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    # ============================================================
    # Финальный ответ (глобальный commit уже не нужен — всё по месту)
    # ============================================================
    successful_files = sum(1 for r in results if r["status"] in ("success", "queued"))
    failed_files = sum(1 for r in results if r["status"] == "failed")

    return {
        "uploaded_files": len(results),
        "successful": successful_files,
        "failed": failed_files,
        "results": results,
        "case_ids": case_ids_map or None,
    }
