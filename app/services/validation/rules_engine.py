# app/services/validation/rules_engine.py
import re
from typing import List, Dict, Any, Tuple

PERSON_RX = re.compile(r"\b([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]\.){1,2}|[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2})\b")
DATE_RX   = re.compile(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
MONEY_RX  = re.compile(r"(?:(\d{1,3}(?:\s?\d{3})+|\d+)(?:[.,]\d{1,2})?)\s?(?:тг|тенге|KZT|₸)", re.IGNORECASE)

def check_text_consistency(text: str) -> Dict[str, Any]:
    """Лёгкая валидация готовых разделов (даты/суммы/персоны упомянуты, если заявлены)."""
    persons = PERSON_RX.findall(text)
    dates   = DATE_RX.findall(text)
    money   = MONEY_RX.findall(text)
    return {
        "has_persons": bool(persons),
        "has_dates":   bool(dates),
        "has_money":   bool(money),
        "counts": {"persons": len(persons), "dates": len(dates), "money": len(money)},
        "warnings": []
    }

def require_inline_citations(text: str) -> Tuple[bool, List[str]]:
    """
    Требуем инлайн-ссылки вида [file_id:page:chunk_id?] или [file_id:page].
    Возвращаем (ok, warnings).
    """
    # Разрешим 2 формата: [xxx:12:abc123] и [xxx:12]
    CIT_RX = re.compile(r"\[([a-zA-Z0-9\-_]+):(\d+)(?::([a-zA-Z0-9\-_]+))?\]")
    citations = CIT_RX.findall(text)
    if not citations:
        return False, ["Нет инлайн-ссылок на источники в тексте (ожидаем формат [fileId:page(:chunkId)])."]
    return True, []
