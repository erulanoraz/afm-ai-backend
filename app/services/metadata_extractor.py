import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# ==========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================

def _detect_language(text: str) -> Optional[str]:
    """Очень грубое определение языка: ru / kk / en / mixed."""
    if not text:
        return None

    cyrillic = sum(1 for ch in text if "А" <= ch <= "я" or ch in "ЁёІіҒғҚқҢңҰұҮүҺһӨөӘә")
    latin = sum(1 for ch in text if "A" <= ch <= "z")

    if cyrillic > 0 and latin == 0:
        # рус / каз — не делим, просто 'cyrillic'
        return "cyrillic"
    if latin > 0 and cyrillic == 0:
        return "latin"
    if cyrillic > 0 and latin > 0:
        return "mixed"

    return None


def _extract_first_match(patterns: List[re.Pattern], text: str) -> Optional[str]:
    for p in patterns:
        m = p.search(text)
        if m:
            return m.group(1)
    return None


def _extract_all_matches(pattern: re.Pattern, text: str, max_items: int = 10) -> List[str]:
    results = []
    for m in pattern.finditer(text):
        val = m.group(0)
        if val not in results:
            results.append(val)
        if len(results) >= max_items:
            break
    return results


# ==========================
# ОСНОВНАЯ ФУНКЦИЯ
# ==========================

def extract_metadata(
    filename: str,
    file_bytes: bytes,
    text_hint: Optional[str] = None,
    sample_size: int = 8192,
) -> Dict[str, Any]:
    """
    Лёгкий rule-based extractor для первичных метаданных документа.

    Работает:
    - по имени файла
    - по первым байтам файла (если это текст)
    - по text_hint (если его передали выше по пайплайну)

    НЕ вызывает LLM, не делает OCR.
    """
    metadata: Dict[str, Any] = {}

    # -----------------------------------------
    # 1) Базовая информация по имени файла
    # -----------------------------------------
    fn_lower = filename.lower()

    metadata["filename"] = filename

    # Простейшее извлечение case_id / больших номеров из имени файла
    case_id_match = re.search(r"(\d{9,})", fn_lower)
    if case_id_match:
        metadata.setdefault("possible_numbers", [])
        num = case_id_match.group(1)
        if num not in metadata["possible_numbers"]:
            metadata["possible_numbers"].append(num)

    # -----------------------------------------
    # 2) Подготавливаем текст для анализа
    # -----------------------------------------
    text_sources: List[str] = []

    if text_hint:
        text_sources.append(text_hint)

    # пытаемся сделать семпл текста из байт (на случай текстовых файлов / PDF с текстом)
    if file_bytes:
        try:
            sample = file_bytes[:sample_size].decode("utf-8", errors="ignore")
            if sample.strip():
                text_sources.append(sample)
        except Exception:
            # если не декодится — пропускаем
            pass

    # если текста нет вообще — возвращаем только filename / possible_numbers
    if not text_sources:
        return metadata

    # объединяем в один текст для простых regex
    merged_text = "\n".join(text_sources)

    # -----------------------------------------
    # 3) Язык
    # -----------------------------------------
    lang = _detect_language(merged_text)
    if lang:
        metadata["language"] = lang

    # -----------------------------------------
    # 4) Даты документа
    # -----------------------------------------
    date_patterns = [
        re.compile(r"\b(\d{2}[./-]\d{2}[./-]\d{4})\b"),  # 12.03.2024 / 12-03-2024
        re.compile(
            r"\b(\d{1,2}\s+"
            r"(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
            r"\s+\d{4}\s*г(?:ода)?)",
            re.IGNORECASE,
        ),
    ]

    doc_date = _extract_first_match(date_patterns, merged_text)
    if doc_date:
        metadata["document_date"] = doc_date

    # -----------------------------------------
    # 5) Номера КУИ / ЕРДР / прочие
    # -----------------------------------------
    kui_patterns = [
        re.compile(r"КУИ\s*№\s*([0-9\-]+)", re.IGNORECASE),
        re.compile(r"КУИ\s*No\.?\s*([0-9\-]+)", re.IGNORECASE),
    ]
    erdr_patterns = [
        re.compile(r"ЕРДР\s*№\s*([0-9\-]+)", re.IGNORECASE),
        re.compile(r"Е[РР]Д[РР]\s*№\s*([0-9\-]+)", re.IGNORECASE),
    ]
    generic_doc_num_patterns = [
        re.compile(r"№\s*([0-9]{6,})"),
    ]

    kui_number = _extract_first_match(kui_patterns, merged_text)
    if kui_number:
        metadata["kui_number"] = kui_number

    erdr_number = _extract_first_match(erdr_patterns, merged_text)
    if erdr_number:
        metadata["erdr_number"] = erdr_number

    doc_number = _extract_first_match(generic_doc_num_patterns, merged_text)
    if doc_number and "document_number" not in metadata:
        metadata["document_number"] = doc_number

    # -----------------------------------------
    # 6) Возможные ФИО (очень грубо)
    # -----------------------------------------
    # шаблон типа "Иванов И.И." или "Иванов Иван Иванович"
    fio_pattern = re.compile(
        r"\b([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2}\s*(?:[А-ЯЁ]\.[А-ЯЁ]\.)?)\b"
    )

    persons = _extract_all_matches(fio_pattern, merged_text, max_items=20)
    if persons:
        metadata["possible_persons"] = persons

    # -----------------------------------------
    # 7) Возможные суммы (тенге / руб / $ / USDT)
    # -----------------------------------------
    amount_pattern = re.compile(
        r"\b\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d+)?\s*(?:тенге|тг|₸|руб(?:лей|\.?)?|₽|usd|\$|usdt)\b",
        re.IGNORECASE,
    )
    amounts = _extract_all_matches(amount_pattern, merged_text, max_items=20)
    if amounts:
        metadata["possible_amounts"] = amounts

    # -----------------------------------------
    # 8) Возможные счета / карты / кошельки
    # -----------------------------------------
    # Примеры: KZ..., 16-20 цифр подряд, USDT адреса (очень грубо)
    account_pattern = re.compile(r"\bKZ[0-9A-Z]{10,}\b")
    card_pattern = re.compile(r"\b\d{4}[ \-]?\d{4}[ \-]?\d{4}[ \-]?\d{4}\b")

    accounts = _extract_all_matches(account_pattern, merged_text, max_items=20)
    cards = _extract_all_matches(card_pattern, merged_text, max_items=20)

    if accounts:
        metadata["possible_accounts"] = accounts
    if cards:
        metadata["possible_cards"] = cards

    # -----------------------------------------
    # 9) Типовые маркеры документа (подсказка для document_classifier)
    # -----------------------------------------
    markers = []
    for kw in ["протокол допроса", "рапорт", "постановление", "заявление", "выписка", "договор"]:
        if re.search(kw, merged_text, re.IGNORECASE):
            markers.append(kw)
    if markers:
        metadata["content_markers"] = markers

    logger.debug(f"📑 extract_metadata({filename}): {metadata}")
    return metadata
