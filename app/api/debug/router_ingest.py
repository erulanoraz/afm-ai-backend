from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session
import uuid
import os
import tempfile

from app.db import get_db
from app.services.ingest_service import process_any_file
from app.db.models import File as DBFile        # ← ВАЖНО: переименовали!
from app.storage.s3_client import upload_to_s3

router = APIRouter(
    prefix="/debug/ingest",
    tags=["DEBUG – Ingest"]
)


@router.post("/")
async def debug_ingest(
    case_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # -----------------------------------------
    # 1. Читаем байты
    # -----------------------------------------
    file_bytes = await file.read()
    filename = file.filename

    # -----------------------------------------
    # 2. Сохраняем файл во временную директорию
    # -----------------------------------------
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"debug_{uuid.uuid4()}_{filename}")

    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    # -----------------------------------------
    # 3. Создаём запись в БД
    # -----------------------------------------
    file_id = uuid.uuid4()
    s3_key = upload_to_s3(file_bytes, filename)

    file_obj = DBFile(
        file_id=file_id,
        filename=filename,
        s3_key=s3_key,
        case_id=case_id,
        uploader="debug",
        metadata={},
    )

    db.add(file_obj)
    db.commit()
    db.refresh(file_obj)

    # -----------------------------------------
    # 4. Запускаем INGEST → OCR → Chunker → Evidence
    # -----------------------------------------
    chunks_created = process_any_file(
        file_path=temp_path,
        file_id=file_id,
        db=db
    )

    # -----------------------------------------
    # 5. Удаляем временный файл
    # -----------------------------------------
    try:
        os.remove(temp_path)
    except:
        pass

    # -----------------------------------------
    # 6. Возвращаем результат
    # -----------------------------------------
    return {
        "file_id": str(file_id),
        "filename": filename,
        "case_id": case_id,
        "s3_key": s3_key,
        "chunks_created": chunks_created,
        "message": "Ingest (SMART OCR 5.1 + Chunker 5.1 + Evidence) completed"
    }
