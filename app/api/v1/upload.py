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
from app.services.chunker import process_pdf_with_smart_ocr, process_text_into_chunks
from app.services.ocr_worker import extract_text_from_pdf  # OCR fallback
from app.services.retrieval import get_file_docs_for_qualifier
from app.services.agents.ai_qualifier import qualify_documents
from app.services.validation.verifier import run_full_verification
from app.services.parser import extract_text_from_file
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
    if not name:
        return None
    m = re.search(r"(\d{15})", name)
    return m.group(1) if m else None


def _validate_file_size(file: UploadFile) -> None:
    if hasattr(file, 'size') and file.size:
        if file.size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Файл {file.filename} слишком большой ({file.size / 1024 / 1024:.1f} МБ). "
                       f"Максимальный размер: {MAX_FILE_SIZE_MB} МБ"
            )

# ============================================================
# Основной endpoint
# ============================================================

@router.post("/")
async def upload_files(
    files: List[UploadFile] = FastAPIFile(...),
    db: Session = Depends(get_db),
):
    results: List[dict] = []
    case_ids_map: dict = {}

    logger.info(f"📤 Начало загрузки {len(files)} файлов")

    for file in files:
        temp_path = None
        try:
            # 1️⃣ Проверка размера
            try:
                _validate_file_size(file)
            except HTTPException as e:
                logger.warning(f"⚠️ {e.detail}")
                results.append({
                    "file_id": None,
                    "filename": file.filename,
                    "chunks_created": 0,
                    "error": e.detail,
                    "status": "failed"
                })
                continue

            # 2️⃣ Временное сохранение файла
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
                temp_path = tmp.name
                content = await file.read()
                if len(content) > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Файл {file.filename} слишком большой. Максимум: {MAX_FILE_SIZE_MB} МБ"
                    )
                tmp.write(content)

            logger.info(f"📥 Загружен файл: {file.filename} ({len(content) / 1024:.1f} КБ)")
            ext = os.path.splitext(file.filename)[1].lower()

            # ============================================================
            # 3️⃣ ZIP
            # ============================================================
            if ext == ".zip":
                extract_dir = tempfile.mkdtemp(prefix="unzipped_")
                try:
                    with zipfile.ZipFile(temp_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                    logger.info(f"📦 Распакован архив {file.filename} → {extract_dir}")
                except zipfile.BadZipFile:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Файл {file.filename} повреждён или не является ZIP архивом"
                    )

                outer_case_id = _extract_case_id_from_name(file.filename)
                zip_inner_ids: List[str] = []

                for root, _, inner_files in os.walk(extract_dir):
                    for inner_name in inner_files:
                        inner_path = os.path.join(root, inner_name)
                        inner_ext = os.path.splitext(inner_name)[1].lower()
                        if inner_ext not in [".pdf", ".docx", ".txt"]:
                            logger.debug(f"⏭️ Пропуск файла: {inner_name}")
                            continue

                        inner_file_id = uuid.uuid4()
                        chunks_created = 0
                        with db.begin_nested():
                            try:
                                detected_case = _extract_case_id_from_name(inner_name)
                                case_id_to_save = detected_case or outer_case_id
                                new_file = File(
                                    file_id=inner_file_id,
                                    filename=inner_name,
                                    case_id=case_id_to_save,
                                    s3_key=f"s3://afm-originals/{inner_name}",
                                    ocr_confidence=0.9,
                                )
                                db.add(new_file)
                                db.flush()

                                if inner_ext == ".pdf":
                                    chunks_created = process_pdf_with_smart_ocr(inner_path, inner_file_id, db)
                                else:
                                    text = extract_text_from_file(inner_path) or ""
                                    if not text.strip():
                                        raise ValueError("Пустой текст")
                                    chunks_created = process_text_into_chunks(inner_file_id, text, db)

                                if hasattr(new_file, "chunks_count"):
                                    new_file.chunks_count = chunks_created

                                logger.info(f"✅ {inner_name}: {chunks_created} чанков")

                                if case_id_to_save:
                                    case_ids_map.setdefault(case_id_to_save, []).append(str(inner_file_id))

                                results.append({
                                    "file_id": str(inner_file_id),
                                    "filename": inner_name,
                                    "chunks_created": chunks_created,
                                    "s3_key": f"s3://afm-originals/{inner_name}",
                                    "case_id": case_id_to_save,
                                    "status": "success"
                                })
                                zip_inner_ids.append(str(inner_file_id))
                            except ValueError as ve:
                                db.rollback()
                                logger.debug(f"⏭️ {inner_name} пропущен: {ve}")
                            except Exception as e:
                                db.rollback()
                                err = f"Ошибка обработки {inner_name}: {str(e)}"
                                logger.error(f"❌ {err}")
                                results.append({
                                    "file_id": str(inner_file_id),
                                    "filename": inner_name,
                                    "chunks_created": 0,
                                    "error": err,
                                    "status": "failed"
                                })

                # итог по ZIP
                try:
                    zip_chunks_total = sum(
                        r.get("chunks_created", 0)
                        for r in results if r.get("file_id") in zip_inner_ids
                    )
                    results.append({
                        "file_id": None,
                        "filename": file.filename,
                        "type": "zip_summary",
                        "chunks_created": zip_chunks_total,
                        "files_processed": len(zip_inner_ids),
                        "case_id": outer_case_id,
                        "status": "success"
                    })
                except Exception as e:
                    logger.error(f"❌ Ошибка агрегации ZIP {file.filename}: {e}")
                finally:
                    shutil.rmtree(extract_dir, ignore_errors=True)

                continue

            # ============================================================
            # 4️⃣ PDF
            # ============================================================
            if ext == ".pdf":
                file_id = uuid.uuid4()
                chunks_created = 0
                with db.begin_nested():
                    try:
                        single_case_id = _extract_case_id_from_name(file.filename)
                        new_file = File(
                            file_id=file_id,
                            filename=file.filename,
                            case_id=single_case_id,
                            s3_key=f"s3://afm-originals/{file.filename}",
                            ocr_confidence=0.9,
                        )
                        db.add(new_file)
                        db.flush()
                        chunks_created = process_pdf_with_smart_ocr(temp_path, file_id, db)
                        if hasattr(new_file, "chunks_count"):
                            new_file.chunks_count = chunks_created
                        logger.info(f"✅ PDF {file.filename}: {chunks_created} чанков")

                        if single_case_id:
                            case_ids_map.setdefault(single_case_id, []).append(str(file_id))

                        results.append({
                            "file_id": str(file_id),
                            "filename": file.filename,
                            "chunks_created": chunks_created,
                            "s3_key": f"s3://afm-originals/{file.filename}",
                            "case_id": single_case_id,
                            "status": "success"
                        })
                    except Exception as e:
                        db.rollback()
                        err = f"Ошибка обработки PDF {file.filename}: {str(e)}"
                        logger.error(f"❌ {err}")
                        results.append({
                            "file_id": str(file_id),
                            "filename": file.filename,
                            "chunks_created": 0,
                            "error": err,
                            "status": "failed"
                        })
                continue

            # ============================================================
            # 5️⃣ DOCX/TXT
            # ============================================================
            if ext in [".docx", ".txt"]:
                file_id = uuid.uuid4()
                chunks_created = 0
                with db.begin_nested():
                    try:
                        single_case_id = _extract_case_id_from_name(file.filename)
                        new_file = File(
                            file_id=file_id,
                            filename=file.filename,
                            case_id=single_case_id,
                            s3_key=f"s3://afm-originals/{file.filename}",
                            ocr_confidence=0.95,
                        )
                        db.add(new_file)
                        db.flush()
                        text = extract_text_from_file(temp_path) or ""
                        if not text.strip():
                            raise ValueError("Пустой текст")
                        chunks_created = process_text_into_chunks(file_id, text, db)
                        if hasattr(new_file, "chunks_count"):
                            new_file.chunks_count = chunks_created
                        logger.info(f"✅ {ext.upper()} {file.filename}: {chunks_created} чанков")

                        if single_case_id:
                            case_ids_map.setdefault(single_case_id, []).append(str(file_id))

                        results.append({
                            "file_id": str(file_id),
                            "filename": file.filename,
                            "chunks_created": chunks_created,
                            "s3_key": f"s3://afm-originals/{file.filename}",
                            "case_id": single_case_id,
                            "status": "success"
                        })
                    except ValueError as ve:
                        db.rollback()
                        logger.debug(f"⏭️ {file.filename} пропущен: {ve}")
                    except Exception as e:
                        db.rollback()
                        err = f"Ошибка обработки {file.filename}: {str(e)}"
                        logger.error(f"❌ {err}")
                        results.append({
                            "file_id": str(file_id),
                            "filename": file.filename,
                            "chunks_created": 0,
                            "error": err,
                            "status": "failed"
                        })
                continue

            # ============================================================
            # 6️⃣ Неподдерживаемый формат
            # ============================================================
            logger.warning(f"⛔ Неподдерживаемый формат: {ext}")
            results.append({
                "file_id": None,
                "filename": file.filename,
                "chunks_created": 0,
                "error": f"Неподдерживаемый формат: {ext}",
                "status": "failed"
            })

        except HTTPException:
            raise
        except Exception as e:
            err = f"Критическая ошибка загрузки {file.filename}: {str(e)}"
            logger.error(f"❌ {err}", exc_info=True)
            results.append({
                "file_id": None,
                "filename": file.filename,
                "chunks_created": 0,
                "error": err,
                "status": "failed"
            })
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.debug(f"🧹 Удалён временный файл: {temp_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось удалить {temp_path}: {e}")

    # ✅ Единый общий коммит после всех файлов
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Ошибка финального коммита: {e}", exc_info=True)

    # ============================================================
    # 7️⃣ Квалификация
    # ============================================================
    qualification_results = []
    if case_ids_map:
        logger.info(f"🤖 Запуск квалификации для {len(case_ids_map)} дел")
        for case_id, file_ids in case_ids_map.items():
            try:
                logger.info(f"📋 Квалификация дела {case_id} ({len(file_ids)} файлов)")
                docs = get_file_docs_for_qualifier(db, file_ids=file_ids, case_id=case_id)
                if not docs:
                    logger.warning(f"⚠️ Нет документов для дела {case_id}")
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
                    "draft_postanovlenie": qualifier.get("final_postanovlenie")
                        if isinstance(qualifier, dict) else None,
                    "status": "success"
                })
                logger.info(f"✅ Квалификация {case_id} завершена: {qualifier.get('verdict', 'N/A')}")
            except Exception as e:
                err = f"Ошибка квалификации дела {case_id}: {str(e)}"
                logger.error(f"❌ {err}")
                qualification_results.append({
                    "case_id": case_id,
                    "files_analyzed": len(file_ids),
                    "error": err,
                    "status": "failed"
                })

    # ============================================================
    # 8️⃣ Финальный ответ
    # ============================================================
    successful_files = sum(1 for r in results if r.get("status") == "success")
    failed_files = sum(1 for r in results if r.get("status") == "failed")

    logger.info(f"📊 Итог: успешно={successful_files}, ошибок={failed_files}, квалификаций={len(qualification_results)}")

    return {
        "uploaded_files": len(results),
        "successful": successful_files,
        "failed": failed_files,
        "results": results,
        "qualifications": qualification_results or None,
    }
