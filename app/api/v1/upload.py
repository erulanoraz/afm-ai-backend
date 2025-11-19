# app/api/v1/upload.py
import uuid
import os
import tempfile
import zipfile
import shutil
import logging
import re
from typing import List, Optional

from fastapi import APIRouter, Depends, UploadFile, File as FastAPIFile, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.db.models import File
from app.services.chunker import (
    process_any_file,
    process_text_into_chunks
)
from app.services.parser import extract_text_from_file
from app.services.retrieval import get_file_docs_for_qualifier
from app.services.agents.ai_qualifier import qualify_documents
from app.services.validation.verifier import run_full_verification
from app.utils.config import settings

# ============================================================
# Константы
# ============================================================
MAX_FILE_SIZE_MB = getattr(settings, 'MAX_FILE_SIZE_MB', 100)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

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
    Работает и для текстовых PDF, и для DOCX/TXT, и для результатов OCR.
    """
    if not text:
        return None
    # Ищем любую последовательность из 15 цифр
    m = re.search(r"(\d{15})", text)
    return m.group(1) if m else None


def _detect_case_id_for_file(
    file_path: str,
    filename: str,
    outer_case_id: Optional[str] = None,
) -> Optional[str]:
    """
    Универсальный детектор номера дела для одного файла.

    Порядок:
    1) ищем в названии файла;
    2) если не нашли — пытаемся вытащить текст (PDF/DOCX/TXT) и найти там;
    3) если не нашли — используем outer_case_id (для файлов внутри ZIP).
    """
    # 1) По имени файла
    case_id = _extract_case_id_from_name(filename)
    if case_id:
        logger.info(f"🔎 case_id={case_id} найден в названии файла: {filename}")
        return case_id

    # 2) По тексту файла
    try:
        text = extract_text_from_file(file_path) or ""
    except Exception as e:
        logger.warning(f"⚠️ Не удалось извлечь текст из файла {filename} для поиска case_id: {e}")
        text = ""

    if text.strip():
        case_id_from_text = _extract_case_id_from_text(text)
        if case_id_from_text:
            logger.info(f"🔎 case_id={case_id_from_text} найден в тексте файла: {filename}")
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
    if hasattr(file, 'size') and file.size:
        if file.size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Файл {file.filename} слишком большой ({file.size / 1024 / 1024:.1f} МБ). "
                    f"Максимальный размер: {MAX_FILE_SIZE_MB} МБ"
                )
            )


# ============================================================
# Основной endpoint
# ============================================================

@router.post("/")
async def upload_files(
    files: List[UploadFile] = FastAPIFile(...),
    db: Session = Depends(get_db),
):
    results = []
    case_ids_map = {}

    logger.info(f"📤 Загружается файлов: {len(files)}")

    for file in files:
        temp_path = None

        try:
            # 1) Проверка размера
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

            # 2) Временное сохранение файла
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
                temp_path = tmp.name
                content = await file.read()
                if len(content) > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Файл {file.filename} слишком большой. Максимум: {MAX_FILE_SIZE_MB} МБ"
                    )
                tmp.write(content)

            ext = os.path.splitext(file.filename)[1].lower()
            logger.info(f"📥 Загружен {file.filename}, размер {len(content)} байт")

            # ============================================================
            # 3) ZIP
            # ============================================================
            if ext == ".zip":
                extract_dir = tempfile.mkdtemp(prefix="unzipped_")
                try:
                    with zipfile.ZipFile(temp_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                except zipfile.BadZipFile:
                    raise HTTPException(status_code=400, detail="ZIP файл повреждён")

                # case_id, который удалось найти в ИМЕНИ самого ZIP
                outer_case_id = _extract_case_id_from_name(file.filename)
                if outer_case_id:
                    logger.info(f"🔎 outer_case_id={outer_case_id} найден в имени ZIP {file.filename}")
                else:
                    logger.info(f"ℹ️ В имени ZIP {file.filename} номер дела не найден")

                zip_inner_ids = []

                for root, _, inner_files in os.walk(extract_dir):
                    for inner_name in inner_files:
                        inner_path = os.path.join(root, inner_name)
                        inner_ext = os.path.splitext(inner_name)[1].lower()
                        if inner_ext not in [".pdf", ".docx", ".txt"]:
                            continue

                        inner_file_id = uuid.uuid4()
                        with db.begin_nested():
                            try:
                                # 🔥 Новый умный поиск case_id:
                                # 1) имя внутреннего файла
                                # 2) текст/ОCR внутреннего файла
                                # 3) outer_case_id (из имени ZIP)
                                case_id_to_save = _detect_case_id_for_file(
                                    file_path=inner_path,
                                    filename=inner_name,
                                    outer_case_id=outer_case_id,
                                )

                                new_file = File(
                                    file_id=inner_file_id,
                                    filename=inner_name,
                                    case_id=case_id_to_save,
                                    s3_key=f"s3://afm-originals/{inner_name}",
                                    ocr_confidence=0.9,
                                )
                                db.add(new_file)
                                db.flush()

                                chunks_created = 0
                                if inner_ext == ".pdf":
                                    # 🔁 Логика чанкинга не меняется
                                    chunks_created = process_any_file(inner_path, inner_file_id, db)
                                else:
                                    text = extract_text_from_file(inner_path) or ""
                                    if text.strip():
                                        chunks_created = process_text_into_chunks(inner_file_id, text, db)

                                new_file.chunks_count = chunks_created
                                logger.info(f"📄 {inner_name}: создано чанков = {chunks_created}")

                                if case_id_to_save:
                                    case_ids_map.setdefault(case_id_to_save, []).append(str(inner_file_id))

                                results.append({
                                    "file_id": str(inner_file_id),
                                    "filename": inner_name,
                                    "chunks_created": chunks_created,
                                    "case_id": case_id_to_save,
                                    "s3_key": f"s3://afm-originals/{inner_name}",
                                    "status": "success",
                                })

                                zip_inner_ids.append(str(inner_file_id))

                            except Exception as e:
                                logger.error(f"❌ Ошибка файла в ZIP {inner_name}: {e}")
                                results.append({
                                    "file_id": str(inner_file_id),
                                    "filename": inner_name,
                                    "chunks_created": 0,
                                    "error": str(e),
                                    "status": "failed"
                                })

                # Итог по ZIP
                total_chunks = sum(
                    r.get("chunks_created", 0)
                    for r in results
                    if r.get("file_id") in zip_inner_ids
                )
                results.append({
                    "file_id": None,
                    "filename": file.filename,
                    "type": "zip_summary",
                    "files_processed": len(zip_inner_ids),
                    "chunks_created": total_chunks,
                    "case_id": outer_case_id,
                    "status": "success"
                })

                shutil.rmtree(extract_dir, ignore_errors=True)
                continue

            # ============================================================
            # 4) PDF
            # ============================================================
            if ext == ".pdf":
                file_id = uuid.uuid4()
                with db.begin_nested():
                    try:
                        # 🔥 Здесь теперь умный поиск case_id:
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
                            ocr_confidence=0.9,
                        )
                        db.add(new_file)
                        db.flush()

                        # Логика чанкинга не меняется
                        chunks_created = process_any_file(temp_path, file_id, db)
                        new_file.chunks_count = chunks_created
                        logger.info(f"📄 PDF {file.filename}: чанков = {chunks_created}")

                        if case_id_extracted:
                            case_ids_map.setdefault(case_id_extracted, []).append(str(file_id))

                        results.append({
                            "file_id": str(file_id),
                            "filename": file.filename,
                            "chunks_created": chunks_created,
                            "case_id": case_id_extracted,
                            "s3_key": f"s3://afm-originals/{file.filename}",
                            "status": "success",
                        })

                    except Exception as e:
                        logger.error(f"❌ Ошибка PDF {file.filename}: {e}")
                        results.append({
                            "file_id": str(file_id),
                            "filename": file.filename,
                            "chunks_created": 0,
                            "error": str(e),
                            "status": "failed",
                        })
                continue

            # ============================================================
            # 5) DOCX / TXT
            # ============================================================
            if ext in [".docx", ".txt"]:
                file_id = uuid.uuid4()
                with db.begin_nested():
                    try:
                        # 🔥 Аналогично ищем case_id и по имени, и по тексту
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
                            ocr_confidence=0.95,
                        )
                        db.add(new_file)
                        db.flush()

                        text = extract_text_from_file(temp_path) or ""
                        if not text.strip():
                            raise ValueError("Пустой текст")

                        chunks_created = process_text_into_chunks(file_id, text, db)
                        new_file.chunks_count = chunks_created

                        if case_id_extracted:
                            case_ids_map.setdefault(case_id_extracted, []).append(str(file_id))

                        results.append({
                            "file_id": str(file_id),
                            "filename": file.filename,
                            "chunks_created": chunks_created,
                            "case_id": case_id_extracted,
                            "status": "success",
                        })

                    except Exception as e:
                        logger.error(f"❌ Ошибка обработки {file.filename}: {e}")
                        results.append({
                            "file_id": str(file_id),
                            "filename": file.filename,
                            "chunks_created": 0,
                            "error": str(e),
                            "status": "failed",
                        })
                continue

            # ============================================================
            # 6) Неподдерживаемый формат
            # ============================================================
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
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    # ============================================================
    # Глобальный коммит
    # ============================================================
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Ошибка общего коммита: {e}")

    # ============================================================
    # 7) Квалификация (логика не меняется)
    # ============================================================
    qualification_results = []

    if case_ids_map:
        for case_id, file_ids in case_ids_map.items():
            try:
                docs = get_file_docs_for_qualifier(db, file_ids=file_ids, case_id=case_id)
                if not docs:
                    continue

                qualifier = qualify_documents(
                    case_id=case_id,
                    docs=docs,
                    city="г. Павлодар",
                    investigator_line="Следователь СЭР ДЭР по Павлодарской области",
                )

                verification = run_full_verification(qualifier)

                qualification_results.append({
                    "case_id": case_id,
                    "files_analyzed": len(file_ids),
                    "qualifier": qualifier,
                    "verification": verification,
                    "draft_postanovlenie": qualifier.get("final_postanovlenie"),
                    "status": "success",
                })

            except Exception as e:
                qualification_results.append({
                    "case_id": case_id,
                    "files_analyzed": len(file_ids),
                    "error": str(e),
                    "status": "failed",
                })

    # ============================================================
    # 8) Финальный ответ
    # ============================================================
    successful_files = sum(1 for r in results if r["status"] == "success")
    failed_files = sum(1 for r in results if r["status"] == "failed")

    return {
        "uploaded_files": len(results),
        "successful": successful_files,
        "failed": failed_files,
        "results": results,
        "qualifications": qualification_results or None,
    }
