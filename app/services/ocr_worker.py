# app/services/ocr_worker.py

import os
import logging
from io import BytesIO
from typing import Optional, List

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
from PyPDF2 import PdfReader
from PIL import Image

from app.utils.config import settings

logger = logging.getLogger(__name__)
ocr_corr_logger = logging.getLogger("OCR_CORRECTOR")

# ============================================================
# ⚙️ Инициализация Tesseract + Poppler
# ============================================================

pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_PATH
POPPLER_PATH = getattr(settings, "POPPLER_PATH", None)

OCR_LANG = os.getenv("OCR_LANG", "rus+kaz+eng")

OCR_OEM = 1
PSM_CANDIDATES = [6, 4, 3]  # 6 — блок текста, 4 — колонки, 3 — авто


# ============================================================
# 🔧 Вспомогательные функции
# ============================================================

def _normalize_ocr_text(text: str) -> str:
    """
    Лёгкая нормализация OCR-результата:
    - убираем технический мусор
    - чистим множественные пробелы и лишние переносы
    """
    if not text:
        return ""

    t = text.replace("\r", "")

    import re
    t = re.sub(r"-{2,}\s*Page\s*\d+\s*-{2,}", "", t, flags=re.IGNORECASE)

    garbage_patterns = [
        r"сканировано\s*с\s*помощью.*",
        r"©\s*Все права защищены.*",
        r"QR[- ]?код.*",
        r"электронный документ.*",
        r"Документ создан.*",
        r"страница\s*\d+\s*из\s*\d+.*",
    ]
    for g in garbage_patterns:
        t = re.sub(g, "", t, flags=re.IGNORECASE)

    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()


def _preprocess_image(image: Image.Image) -> Image.Image:
    """
    Базовая предобработка для Standard OCR:
    - перевод в оттенки серого
    - лёгкий бинарный трешхолд
    """
    try:
        img = np.array(image)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        _, thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return Image.fromarray(thresh)
    except Exception as e:
        logger.warning(f"⚠️ Ошибка предобработки изображения: {e}")
        return image


# ============================================================
# 🧠 LLM-корректор для OCR (опционально)
# ============================================================

import json
import requests


_OCR_CORRECTOR_ENABLED = bool(getattr(settings, "OCR_CORRECTOR_ENABLED", True))
_OCR_CORRECTOR_URL = getattr(settings, "OCR_CORRECTOR_URL", None) or getattr(
    settings, "LLM_GATEWAY_URL", None
)
_OCR_CORRECTOR_MODEL = getattr(settings, "OCR_CORRECTOR_MODEL", "gpt-4o-mini")
_OCR_CORRECTOR_API_KEY = getattr(settings, "OCR_CORRECTOR_API_KEY", None)


def _correct_ocr_with_llm(raw_text: str, page_num: int) -> str:
    """
    Пост-коррекция OCR через LLM (без перефразирования).
    Только:
      - исправление ошибок распознавания,
      - восстановление пунктуации,
      - склейка разорванных строк.
    """
    if not raw_text:
        return raw_text

    if not (_OCR_CORRECTOR_ENABLED and _OCR_CORRECTOR_URL):
        return raw_text

    try:
        ocr_corr_logger.info(
            f"🧠 OCR_CORRECTOR: page={page_num}, len={len(raw_text)}"
        )

        system_prompt = (
            "Ты помогаешь исправлять результат OCR российских/казахстанских "
            "юридических документов (протоколы, постановления, рапорты). "
            "Исправляй только опечатки и артефакты сканирования, не добавляй "
            "новых слов и не меняй смысл. Сохраняй структуру текста."
        )

        payload = {
            "model": _OCR_CORRECTOR_MODEL,
            "temperature": 0.0,
            "max_tokens": max(512, len(raw_text) // 3),
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Страница: {page_num}\n\n"
                        f"Вот текст после OCR. Очисти его и исправь только ошибки OCR:\n\n"
                        f"{raw_text}"
                    ),
                },
            ],
        }

        headers = {
            "Content-Type": "application/json",
        }
        if _OCR_CORRECTOR_API_KEY:
            headers["Authorization"] = f"Bearer {_OCR_CORRECTOR_API_KEY}"

        resp = requests.post(
            _OCR_CORRECTOR_URL.rstrip("/") + "/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not content:
            return raw_text

        return _normalize_ocr_text(content)

    except Exception as e:
        ocr_corr_logger.error(f"❌ OCR_CORRECTOR ошибка page={page_num}: {e}")
        return raw_text


# ============================================================
# 🧾 OCR по Image
# ============================================================

def run_tesseract_ocr_image(
    image: Image.Image,
    page_num: int = 1,
    use_preprocessing: bool = True,
) -> str:
    """
    Запуск Tesseract по PIL.Image.
    Используется Smart-OCR и debug-эндпоинтами.
    """
    if use_preprocessing:
        image = _preprocess_image(image)

    for psm in PSM_CANDIDATES:
        try:
            config = f"--oem {OCR_OEM} --psm {psm}"
            text = pytesseract.image_to_string(
                image,
                lang=OCR_LANG,
                config=config,
            )
            text = _normalize_ocr_text(text)
            logger.debug(
                f"OCR(page): стр.{page_num}, PSM={psm}, len={len(text)}"
            )

            if len(text.strip()) > 30:
                corrected = _correct_ocr_with_llm(text, page_num)
                return corrected or text
        except Exception as e:
            logger.error(f"❌ Tesseract error page={page_num}, PSM={psm}: {e}")

    return ""


# ============================================================
# 📄 OCR по PDF-странице
# ============================================================

def run_tesseract_ocr(
    file_path: str,
    page_num: int,
    use_preprocessing: bool = True,
) -> str:
    try:
        pages = convert_from_path(
            file_path,
            dpi=300,
            poppler_path=POPPLER_PATH,
            first_page=page_num,
            last_page=page_num,
            fmt="jpeg",
        )

        if not pages:
            logger.warning(f"⚠️ convert_from_path не вернул стр.{page_num}")
            return ""

        image = pages[0]
        text = run_tesseract_ocr_image(
            image=image,
            page_num=page_num,
            use_preprocessing=use_preprocessing,
        )
        return text or ""
    except Exception as e:
        logger.error(f"❌ Ошибка OCR file={file_path}, page={page_num}: {e}")
        return ""


# ============================================================
# 📚 PDF text-layer → OCR fallback
# ============================================================

def _extract_pdf_text_layer(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        pieces: List[str] = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                pieces.append(t)

        full = "\n\n".join(pieces)
        full = _normalize_ocr_text(full)

        if len(full) < 200:
            return ""

        logger.info(
            f"📄 PDF text-layer обнаружен, длина={len(full)} символов "
            f"(страниц={len(reader.pages)})"
        )
        return full
    except Exception as e:
        logger.warning(f"⚠️ Ошибка извлечения text-layer из PDF: {e}")
        return ""


def extract_text_from_pdf(
    file_path: str,
    dpi: int = 300,
    use_preprocessing: bool = True,
) -> str:
    text_layer = _extract_pdf_text_layer(file_path)
    if text_layer:
        return text_layer

    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
    except Exception as e:
        logger.error(f"❌ Не удалось прочитать PDF для OCR: {e}")
        return ""

    logger.info(f"📖 OCR fallback по всему PDF, страниц={total_pages}")

    all_pages: List[str] = []
    for page_num in range(1, total_pages + 1):
        t = run_tesseract_ocr(
            file_path=file_path,
            page_num=page_num,
            use_preprocessing=use_preprocessing,
        )
        if t.strip():
            all_pages.append(t)

    full = "\n\n".join(all_pages)
    return _normalize_ocr_text(full)


# ============================================================
# 🐞 DEBUG OCR
# ============================================================

def debug_ocr_single_page(
    file_path: str,
    page_num: int,
    use_preprocessing: bool = True,
) -> str:
    logger.info(f"🐞 DEBUG OCR: file={file_path}, page={page_num}")
    return run_tesseract_ocr(
        file_path=file_path,
        page_num=page_num,
        use_preprocessing=use_preprocessing,
    )


# ============================================================
# 🖼 OCR Image Mode (для debug / router_chunker)
# ============================================================

def preprocess_image(image_bytes: bytes) -> Image.Image:
    """
    Подготовка изображения к OCR:
    - grayscale
    - adaptive threshold
    - лёгкая резкость
    """
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return Image.open(BytesIO(image_bytes))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(thr, -1, kernel)

    pil_image = Image.fromarray(sharp)
    return pil_image


def ocr_image_bytes(image: Image.Image) -> dict:
    """
    OCR для уже подготовленного PIL.Image.
    Возвращает dict:
    {
        "text": "...",
        "conf": float
    }
    """
    try:
        text = pytesseract.image_to_string(
            image,
            lang=OCR_LANG,
            config="--oem 1 --psm 6"
        )
        return {
            "text": text.strip(),
            "conf": 0.95
        }
    except Exception as e:
        logger.error(f"❌ Ошибка OCR изображения: {e}")
        return {"text": "", "conf": 0.0}


# ============================================================
# 📄 PDF text-layer (bytes) + fallback OCR (bytes)
# ============================================================

def extract_pdf_text_layer(pdf_bytes: bytes) -> list:
    """
    Возвращает:
    [
      {"page": 1, "text": "..."},
      {"page": 2, "text": "..."},
      ...
    ]
    через PyPDF2 по BytesIO.
    """
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        pages = []

        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append({
                "page": i,
                "text": _normalize_ocr_text(text)
            })

        return pages

    except Exception as e:
        logger.error(f"❌ extract_pdf_text_layer ошибка: {e}")
        return []


def extract_pdf_text_fallback(pdf_bytes: bytes, page_num: int) -> str:
    """
    Fallback OCR одной страницы PDF по bytes.
    """
    try:
        images = convert_from_bytes(
            pdf_bytes,
            dpi=300,
            poppler_path=POPPLER_PATH,
            first_page=page_num,
            last_page=page_num,
            fmt="jpeg"
        )
        if not images:
            return ""

        pil_img = images[0]
        text = run_tesseract_ocr_image(
            pil_img,
            page_num=page_num,
            use_preprocessing=True
        )
        return text.strip()

    except Exception as e:
        logger.error(f"❌ extract_pdf_text_fallback ошибка: {e}")
        return ""
