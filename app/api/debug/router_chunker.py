# app/api/debug/router_chunker.py

import os
import magic
from fastapi import APIRouter, UploadFile, File

from app.services.ocr_worker import (
    preprocess_image,
    ocr_image_bytes,
    extract_text_from_pdf,
)
from app.services.chunker import (
    split_sentences,
    chunk_by_sentences,
    build_chunk_payload,
    _normalize_text,
)

router = APIRouter(
    prefix="/debug/chunker",
    tags=["DEBUG – Chunker"]
)

# Универсальная temp-папка (Windows/Linux)
TEMP_DIR = os.path.join(os.getcwd(), "temp_debug")
os.makedirs(TEMP_DIR, exist_ok=True)


@router.post("/", summary="Проверка Chunker (sentence-based)")
async def debug_chunker(
    file: UploadFile = File(...),
    max_chars: int = 1500,
    min_chars: int = 300
):
    """
    Debug Chunker:
    - OCR изображений
    - PDF text-layer
    - sentence splitting
    - simple chunk_by_sentences
    """

    file_bytes = await file.read()
    content_type = magic.Magic(mime=True).from_buffer(file_bytes)

    # -------------------------------------------------------
    # STEP 1 – TEXT / OCR
    # -------------------------------------------------------

    if file.filename.lower().endswith(".pdf"):
        temp_file = os.path.join(TEMP_DIR, "temp_debug.pdf")
        with open(temp_file, "wb") as f:
            f.write(file_bytes)

        text = extract_text_from_pdf(temp_file) or ""

        pages = [{
            "page": 1,
            "text": text
        }]

    elif content_type.startswith("image/"):
        processed = preprocess_image(file_bytes)
        ocr_res = ocr_image_bytes(processed)
        pages = [{
            "page": 1,
            "text": ocr_res.get("text", "")
        }]

    else:
        try:
            text = file_bytes.decode("utf-8")
        except:
            text = file_bytes.decode("latin1")

        pages = [{
            "page": 1,
            "text": text
        }]

    # -------------------------------------------------------
    # STEP 2 — SIMPLE sentence chunker (chunk_by_sentences)
    # -------------------------------------------------------

    result_chunks = []

    for page in pages:
        page_text = _normalize_text(page["text"])

        chunks = chunk_by_sentences(
            text=page_text,
            max_chars=max_chars,
            min_chars=min_chars
        )

        for idx, chunk_text in enumerate(chunks, start=1):
            payload = build_chunk_payload(
                chunk_text=chunk_text,
                page=page["page"],
                section="debug",
                paragraph_index=idx
            )

            result_chunks.append({
                "chunk_index": idx,
                "page": page["page"],
                "text": chunk_text,
                "tokens": len(chunk_text.split()),
                "evidence": payload
            })

    # -------------------------------------------------------
    # RESPONSE
    # -------------------------------------------------------

    return {
        "pages_detected": len(pages),
        "total_chunks": len(result_chunks),
        "chunks": result_chunks
    }
