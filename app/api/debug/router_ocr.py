# app/api/debug/router_ocr.py

from fastapi import APIRouter, UploadFile, File
import magic
import tempfile
import os
from typing import List, Dict, Any
from io import BytesIO
from PIL import Image

from app.services.ocr_worker import (
    run_tesseract_ocr,
    run_tesseract_ocr_image,
)
from app.services.ocr_corrector import correct_ocr_text

router = APIRouter(prefix="/debug/ocr", tags=["DEBUG - OCR"])


@router.post("/")
async def debug_ocr(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    DEBUG OCR:
    - Принимает PDF / image / txt
    - Делает постраничный OCR
    - Делает OCR-Correction (STRICT GPT-OSS)
    - Возвращает raw_text + corrected_text
    """

    file_bytes = await file.read()
    content_type = magic.Magic(mime=True).from_buffer(file_bytes)
    pages: List[Dict[str, Any]] = []

    # =====================================================
    # 1. PDF DOCUMENT
    # =====================================================
    if file.filename.lower().endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            pdf_path = tmp.name

        try:
            max_pages = 50
            empty_streak = 0

            for page_num in range(1, max_pages + 1):

                raw_text = run_tesseract_ocr(pdf_path, page_num=page_num)

                if not raw_text or not raw_text.strip():
                    empty_streak += 1
                    if empty_streak >= 1:
                        break
                    continue

                empty_streak = 0
                corrected = correct_ocr_text(raw_text)

                pages.append(
                    {
                        "page": page_num,
                        "raw_text": raw_text,
                        "text": corrected,
                        "conf": None,
                    }
                )
        finally:
            try:
                os.remove(pdf_path)
            except Exception:
                pass

    # =====================================================
    # 2. IMAGE FILE (jpg, png, etc)
    # =====================================================
    elif content_type.startswith("image/"):
        img = Image.open(BytesIO(file_bytes))
        raw_text = run_tesseract_ocr_image(img)
        corrected = correct_ocr_text(raw_text)

        pages.append(
            {
                "page": 1,
                "raw_text": raw_text,
                "text": corrected,
                "conf": None,
            }
        )

    # =====================================================
    # 3. TEXT FILE (.txt)
    # =====================================================
    else:
        try:
            raw_text = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            raw_text = ""

        corrected = correct_ocr_text(raw_text)

        pages.append(
            {
                "page": 1,
                "raw_text": raw_text,
                "text": corrected,
                "conf": None,
            }
        )

    return {
        "pages": pages,
        "total_pages": len(pages),
        "avg_confidence": None,
    }
