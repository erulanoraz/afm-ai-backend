# app/services/ocr_worker.py

import os
import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
import logging
import tempfile
import cv2
import numpy as np
from app.utils.config import settings

logger = logging.getLogger(__name__)

# ============================================================
# ⚙️ Инициализация Tesseract + Poppler
# ============================================================

pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_PATH
POPPLER_PATH = settings.POPPLER_PATH

OCR_LANG = "rus+kaz"
OCR_CONFIG = "--oem 1 --psm 1"  # LSTM engine + automatic layout


# ============================================================
# 🖼️ Предобработка изображения
# ============================================================

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Улучшает читаемость OCR:
    - grayscale
    - denoise
    - Otsu threshold
    - morphological operations
    - convert back to RGB
    """
    try:
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        denoised = cv2.fastNlMeansDenoising(
            gray, None, h=10, templateWindowSize=7, searchWindowSize=21
        )

        _, binary = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        result = Image.fromarray(binary).convert("RGB")
        return result

    except Exception as e:
        logger.warning(f"⚠️ Ошибка preprocess_image: {e}")
        return image


# ============================================================
# 🔧 Общий OCR по списку страниц (PIL Images)
# ============================================================

def _ocr_pages(
    pages,
    start_page_index: int = 1,
    use_preprocessing: bool = True,
    log_prefix: str = "OCR",
):
    text_blocks = []
    ocr_pages = 0
    empty_pages = 0
    total = len(pages)

    for i, page in enumerate(pages, start=start_page_index):
        try:
            proc = preprocess_image(page) if use_preprocessing else page

            text = pytesseract.image_to_string(
                proc,
                lang=OCR_LANG,
                config=OCR_CONFIG
            )

            if text.strip():
                ocr_pages += 1
                text_blocks.append(f"\n--- Page {i} ---\n{text}")
            else:
                empty_pages += 1
                logger.warning(f"⚠️ {log_prefix}: стр. {i} пустая")

        except Exception as e:
            logger.error(f"❌ OCR ошибка на стр. {i}: {e}")

        finally:
            try: page.close()
            except: pass
            try:
                if proc is not page:
                    proc.close()
            except: pass

    logger.info(
        f"📊 {log_prefix}: страниц={total}, OCR успешных={ocr_pages}, пустых={empty_pages}"
    )

    return "\n".join(text_blocks)


# ============================================================
# 📄 OCR целого PDF файла
# ============================================================

def extract_text_from_pdf(
    file_path: str,
    dpi: int = 300,
    use_preprocessing: bool = True,
) -> str:
    text_blocks = []

    try:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > 50:
            dpi = 200
            logger.warning(f"📉 Большой PDF ({size_mb:.1f}MB) → DPI=200")

        # pdfinfo
        try:
            info = pdfinfo_from_path(file_path, poppler_path=POPPLER_PATH)
            pages = int(info.get("Pages", 0))
            if pages > 500:
                logger.error(f"⛔ PDF {pages} стр. > лимита 500")
                return " "
        except Exception:
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            pages = convert_from_path(
                file_path,
                dpi=dpi,
                poppler_path=POPPLER_PATH,
                output_folder=tmpdir,
                fmt="jpeg",
                thread_count=2,
            )

            full_text = _ocr_pages(
                pages,
                start_page_index=1,
                use_preprocessing=use_preprocessing,
                log_prefix="OCR(full)"
            )
            text_blocks.append(full_text)

        final = "\n".join([t for t in text_blocks if t])

        if not final.strip():
            logger.warning("⚠️ extract_text_from_pdf вернул пусто")
            return " "

        return final

    except Exception as e:
        logger.error(f"❌ extract_text_from_pdf ошибка: {e}", exc_info=True)
        return " "


# ============================================================
# 🧩 OCR одной PDF-страницы (совместимо с chunker.py)
# ============================================================

def run_tesseract_ocr(
    file_path: str,
    page_num: int | None = None,
    use_preprocessing: bool = True,
) -> str:
    """
    Если page_num=None — OCR всего PDF.
    Если page_num задан — OCR конкретной страницы.
    """

    # Полный файл
    if page_num is None:
        return extract_text_from_pdf(file_path, use_preprocessing=use_preprocessing)

    # OCR одной страницы
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            pages = convert_from_path(
                file_path,
                dpi=300,
                poppler_path=POPPLER_PATH,
                output_folder=tmpdir,
                fmt="jpeg",
                first_page=page_num,
                last_page=page_num,
                thread_count=1,
            )

            if not pages:
                logger.warning(f"⚠️ Не удалось извлечь изображение стр. {page_num}")
                return ""

            return _ocr_pages(
                pages,
                start_page_index=page_num,
                use_preprocessing=use_preprocessing,
                log_prefix="OCR(page)"
            )

    except Exception as e:
        logger.error(f"❌ run_tesseract_ocr ошибка на стр. {page_num}: {e}")
        return ""


# ============================================================
# 🖼️ OCR по отдельному изображению (используется SMART OCR)
# ============================================================

def run_tesseract_ocr_image(
    image: Image.Image,
    page_num: int | None = None,
    use_preprocessing: bool = True,
) -> str:
    """
    OCR для PIL.Image. Поддерживает page_num для логов.
    """
    try:
        proc = preprocess_image(image) if use_preprocessing else image

        text = pytesseract.image_to_string(
            proc,
            lang=OCR_LANG,
            config=OCR_CONFIG,
        )

        return text or ""

    except Exception as e:
        logger.error(f"❌ run_tesseract_ocr_image ошибка на стр. {page_num}: {e}")
        return ""

    finally:
        try:
            if proc is not image:
                proc.close()
        except:
            pass


# ============================================================
# 🧪 OCR по байтам PDF (если нужно для ingest_service)
# ============================================================

def ocr_pdf_bytes(pdf_bytes: bytes, dpi: int = 300):
    """
    Возвращает (text, confidence, pages_count)
    """
    confidence = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            tmp_path = tmp.name

        text = extract_text_from_pdf(tmp_path, dpi=dpi)
        try:
            os.remove(tmp_path)
        except:
            pass

        return text, confidence, None

    except Exception as e:
        logger.error(f"❌ ocr_pdf_bytes ошибка: {e}")
        return "", None, None


# ============================================================
# 🖼️ OCR по изображению (bytes → PIL → текст)
# ============================================================

def ocr_image_bytes(img_bytes: bytes):
    try:
        img = Image.open(tempfile.SpooledTemporaryFile())
        img.file = img_bytes
        text = run_tesseract_ocr_image(img)
        return text, None
    except Exception as e:
        logger.error(f"❌ ocr_image_bytes ошибка: {e}")
        return "", None
