# app/services/parser.py
from docx import Document
import logging

logger = logging.getLogger(__name__)


def extract_text_from_file(path: str) -> str:
    """
    Универсальный парсер текстовых форматов.
    PDF НЕ парсится здесь — он обрабатывается в OCR_worker.
    Возвращает чистый текст.
    """
    try:
        ext = path.lower().split(".")[-1]

        # PDF → отдаём обработку OCR_worker / chunker
        if ext == "pdf":
            return ""  # сигнал для process_any_file: нужен OCR/Smart-OCR

        # DOCX
        elif ext == "docx":
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs).strip()

        # TXT
        elif ext == "txt":
            try:
                with open(path, encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                # fallback под windows-1251
                with open(path, encoding="cp1251", errors="ignore") as f:
                    return f.read().strip()

        else:
            logger.warning(f"⚠️ Неподдерживаемое расширение: {ext}")
            return ""

    except Exception as e:
        logger.error(f"Ошибка извлечения текста из {path}: {e}")
        return ""
