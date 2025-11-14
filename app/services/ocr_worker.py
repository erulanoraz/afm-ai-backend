import os
import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
import logging
import tempfile
import cv2
import numpy as np
from app.utils.config import settings  # ✅ подключаем .env настройки

logger = logging.getLogger(__name__)

# ============================================================
# ⚙️ Инициализация путей (из .env)
# ============================================================
pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_PATH
POPPLER_PATH = settings.POPPLER_PATH

# Общие настройки OCR
OCR_LANG = "rus+kaz"
OCR_CONFIG = "--oem 1 --psm 1"  # LSTM + авто-лейаут


# ============================================================
# 🖼️ Image Preprocessing (бинаризация и улучшение контраста)
# ============================================================
def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Предварительная обработка изображения для улучшения OCR:
    • Конвертация в grayscale
    • Binary threshold (Otsu)
    • Удаление шума (морфологические операции)
    • Увеличение контраста
    """
    try:
        # PIL → OpenCV (BGR)
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Denoise (убирает шум)
        denoised = cv2.fastNlMeansDenoising(
            gray, None, h=10, templateWindowSize=7, searchWindowSize=21
        )

        # Binary threshold (Otsu автоматически подбирает порог)
        _, binary = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Морфологические операции (закрытие + открытие)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # OpenCV → PIL; приводим к RGB, Tesseract такое любит
        result = Image.fromarray(binary).convert("RGB")
        logger.debug("✅ Image preprocessing завершён")

        return result

    except Exception as e:
        logger.warning(
            f"⚠️ Image preprocessing ошибка: {e}, возвращаю исходное изображение"
        )
        return image


# ============================================================
# 🔧 Вспомогательная функция: OCR по списку страниц (PIL.Image)
# ============================================================
def _ocr_pages(
    pages,
    start_page_index: int = 1,
    use_preprocessing: bool = True,
    log_prefix: str = "OCR",
) -> str:
    """
    Общая логика OCR для списка страниц (PIL.Image).
    • Гарантирует закрытие Image-объектов
    • Используется и в extract_text_from_pdf, и в run_tesseract_ocr(page_num)
    """
    text_blocks = []
    ocr_pages, empty_pages, total_pages = 0, 0, len(pages)

    for i, page in enumerate(pages, start=start_page_index):
        processed_page = page
        try:
            # 🖼️ Image preprocessing (бинаризация)
            if use_preprocessing:
                processed_page = preprocess_image(page)

            # 🧠 OCR с OEM=1 (LSTM engine - лучший для качества)
            txt = pytesseract.image_to_string(
                processed_page,
                lang=OCR_LANG,
                config=OCR_CONFIG,
            )

            if txt.strip():
                ocr_pages += 1
                text_blocks.append(f"\n--- Page {i} ---\n{txt}")
                logger.debug(f"✅ {log_prefix}: страница {i}: {len(txt)} символов")
            else:
                empty_pages += 1
                logger.warning(f"⚠️ {log_prefix}: страница {i}: текст не найден")

        except Exception as e:
            logger.error(f"❌ {log_prefix}: ошибка OCR на странице {i}: {e}")
        finally:
            # 🧹 Закрываем объекты Image (предотвращает утечки)
            try:
                page.close()
            except Exception:
                pass
            if processed_page is not page:
                try:
                    processed_page.close()
                except Exception:
                    pass

    full_text = "\n".join(text_blocks)
    logger.info(
        f"📊 {log_prefix}-итог: страниц={total_pages}, успешно={ocr_pages}, пустых={empty_pages}"
    )
    return full_text


# ============================================================
# 📄 OCR обработка PDF с защитой памяти и оптимизацией
# ============================================================
def extract_text_from_pdf(
    file_path: str,
    dpi: int = 300,
    use_preprocessing: bool = True,
) -> str:
    text_blocks = []

    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > 50:
            dpi = 200
            logger.warning(f"⚠️ Большой файл ({file_size_mb:.1f} MB), DPI понижен до {dpi}")

        try:
            info = pdfinfo_from_path(file_path, poppler_path=POPPLER_PATH)
            total_pages_info = int(info.get("Pages", 0))
            if total_pages_info > 500:
                logger.error(
                    f"⛔ PDF содержит {total_pages_info} страниц — превышен лимит (500). OCR пропущен."
                )
                return " "
        except Exception as e:
            logger.debug(f"ℹ️ Не удалось получить pdfinfo: {e}")

        logger.info(f"📊 OCR параметры: DPI={dpi}, OEM=1, preprocessing={'ON' if use_preprocessing else 'OFF'}")

        with tempfile.TemporaryDirectory() as temp_dir:
            pages = convert_from_path(
                file_path,
                dpi=dpi,
                poppler_path=POPPLER_PATH,
                output_folder=temp_dir,
                fmt="jpeg",
                thread_count=2,
            )

            total_pages = len(pages)
            logger.info(f"📄 OCR: конвертировано {total_pages} страниц ({os.path.basename(file_path)})")

            full_text = _ocr_pages(
                pages,
                start_page_index=1,
                use_preprocessing=use_preprocessing,
                log_prefix="OCR",
            )
            text_blocks.append(full_text)

        full_text = "\n".join([t for t in text_blocks if t])

        if not full_text.strip():
            logger.warning(f"⚠️ OCR не извлёк текст из {file_path}")
            return " "  # ⚡ минимальный текст для chunker

        return full_text

    except Exception as e:
        logger.error(f"❌ Ошибка OCR при обработке {file_path}: {e}", exc_info=True)
        return " "  # ⚡ fail-safe guaranteed



def run_tesseract_ocr(
    file_path: str,
    page_num: int | None = None,
    use_preprocessing: bool = True,
) -> str:
    """
    Совместимая обёртка для вызова OCR.

    ✅ Важно:
    • Если page_num is None → OCR всего файла (как раньше).
    • Если page_num задан → OCR ТОЛЬКО этой страницы.
      Это полностью совместимо с текущим chunker.py:
      run_tesseract_ocr(file_path, page_num=i)
    """
    # Вариант 1: полный файл
    if page_num is None:
        return extract_text_from_pdf(file_path, use_preprocessing=use_preprocessing)

    # Вариант 2: одна страница (используется SMART OCR в chunker.py)
    try:
        logger.info(f"📄 OCR одной страницы PDF: {os.path.basename(file_path)}, page={page_num}")

        with tempfile.TemporaryDirectory() as temp_dir:
            pages = convert_from_path(
                file_path,
                dpi=300,
                poppler_path=POPPLER_PATH,
                output_folder=temp_dir,
                fmt="jpeg",
                first_page=page_num,
                last_page=page_num,
                thread_count=1,
            )

            if not pages:
                logger.warning(f"⚠️ Не удалось получить изображение страницы {page_num}")
                return ""

            # здесь всего 1 страница, но используем общий helper
            text = _ocr_pages(
                pages,
                start_page_index=page_num,
                use_preprocessing=use_preprocessing,
                log_prefix="OCR(page)",
            )
            return text

    except Exception as e:
        logger.error(
            f"❌ Ошибка OCR страницы PDF (page={page_num}) в run_tesseract_ocr: {e}",
            exc_info=True,
        )
        return ""


# ============================================================
# 🚀 Опционально: асинхронная очередь для Celery
# ============================================================
# Если ты хочешь сделать OCR-обработку асинхронной (например, для 100 файлов),
# добавь Celery worker и зарегистрируй задачу:
#
# from app.celery_app import celery
#
# @celery.task(name="ocr.extract_text")
# def extract_text_task(file_path: str):
#     return extract_text_from_pdf(file_path)
#
# После этого можно вызывать: extract_text_task.delay(file_path)
# ============================================================
# ============================================================
# 📌 Совместимость с chunker.py (OCR по изображению)
# ============================================================
def run_tesseract_ocr_image(
    image: Image.Image,
    page_num: int | None = None,
    use_preprocessing: bool = True,
) -> str:
    """
    Вызов OCR для одного изображения (PIL.Image).
    Это используется SMART OCR в chunker.py

    Улучшения:
    • Image preprocessing с бинаризацией
    • OEM=1 для лучшего качества
    • Поддержка rus+kaz
    • Корректное закрытие временных объектов
    """
    processed_image = image
    try:
        # 🖼️ Preprocessing для улучшения качества
        if use_preprocessing:
            processed_image = preprocess_image(image)

        # 🧠 OCR с OEM=1 (LSTM engine)
        text = pytesseract.image_to_string(
            processed_image,
            lang=OCR_LANG,
            config=OCR_CONFIG,
        )

        return text
    except Exception as e:
        logger.error(f"⚠️ Ошибка Tesseract OCR (page {page_num}): {e}")
        return ""
    finally:
        # Обычно image приходит извне и его закрывать не нужно,
        # но если мы создали отдельную копию (processed_image), её лучше закрыть.
        if processed_image is not image:
            try:
                processed_image.close()
            except Exception:
                pass
