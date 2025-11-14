# app/services/parser.py
from pdfminer.high_level import extract_text
from docx import Document
import logging

logger = logging.getLogger(__name__)

def extract_text_from_file(path: str) -> str:
    """
    Универсальный парсер: PDF, DOCX, TXT.
    Возвращает чистый текст.
    """
    try:
        ext = path.lower().split(".")[-1]
        if ext == "pdf":
            return extract_text(path)
        elif ext == "docx":
            doc = Document(path)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext == "txt":
            with open(path, encoding="utf-8", errors="ignore") as f:
                return f.read()
        else:
            logger.warning(f"⚠️ Неподдерживаемое расширение: {ext}")
            return ""
    except Exception as e:
        logger.error(f"Ошибка извлечения текста из {path}: {e}")
        return ""
