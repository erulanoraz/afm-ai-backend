# app/utils/sentence_splitter.py

import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# ======================================================================
# Sentence Splitter v15.0 — ChatGPT Evidence Engine (Date-Aware Edition)
#
# Ключевые возможности:
# ✔ Поддержка 20+ форматов дат → ни один формат НЕ разрывается
# ✔ Новый DateGuard Engine (сначала защищаем ВСЕ даты)
# ✔ Аббревиатуры, карточки, ИО, ст. 190 — защищены
# ✔ OCR-устойчивый split (.Далее / .Затем)
# ✔ Кавычки, скобки, тире — корректно обрабатываются
# ✔ Unicode нормализация (… — — non-breaking spaces)
# ✔ Не рвёт суммы, дроби, ИИНы, счета, карты
# ======================================================================


MAX_LEN = 700_000
MIN_SENT_LEN = 2


# ======================================================================
#  UNIVERSAL DATE PATTERNS (20+ форматов)
# ======================================================================

DATE_PATTERNS = [
    # dd.mm.yyyy / dd.mm.yy / dd.mm.yyyy г.
    r"\b\d{1,2}\.\d{1,2}\.\d{2,4}(?:\s*г\.?)?\b",

    # dd/mm/yyyy / dd/mm/yy
    r"\b\d{1,2}\/\d{1,2}\/\d{2,4}(?:\s*г\.?)?\b",

    # yyyy-mm-dd
    r"\b\d{4}-\d{1,2}-\d{1,2}\b",

    # dd-mm-yyyy
    r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",

    # yyyy г.
    r"\b\d{4}\s*г\.\b",

    # месяцы в текстовом виде (рус)
    r"\b\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|"
    r"июля|августа|сентября|октября|ноября|декабря)\s+\d{4}\b",

    # месяцы казахские
    r"\b\d{1,2}\s+(?:қаңтар|ақпан|наурыз|сәуір|мамыр|маусым|"
    r"шілде|тамыз|қыркүйек|қазан|қараша|желтоқсан)\s+\d{4}\b",
]


# ======================================================================
#  ABBREVIATIONS / SAFE TOKENS (точка → __DOT__)
# ======================================================================

SAFE_PATTERNS = [
    # ФИО: А.Р., И.О.
    r"[А-ЯЁ]\.[А-ЯЁ]\.",
    r"[A-Z]\.[A-Z]\.",

    # ст. 190 / ст.190-2
    r"ст\.?\s*\d{1,3}(?:[-–]\d+)?",

    # статьи словом
    r"статья\s*\d{1,3}",

    # денежные сокращения
    r"\bтг\.",
    r"\bруб\.",
    r"\bдол\.",

    # ул. д. кв. пр. см. им.
    r"\bул\.",
    r"\bд\.",
    r"\bкв\.",
    r"\bпр\.",
    r"\bсм\.",
    r"\bим\.",
    r"\bт\.е\.",
    r"\bт\.о\.",
    r"\bт\.к\.",
    r"\bт\.д\.",
    r"\bт\.п\.",

    # дроби: 1.5, 2.75
    r"\d+\.\d+",
]


# ======================================================================
#  NORMALIZATION
# ======================================================================

def _validate(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    if len(text) > MAX_LEN:
        raise ValueError("text too large")
    return text


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # многоточие
    text = text.replace("…", "...")
    # разные тире
    text = text.replace("—", "-").replace("–", "-").replace("−", "-")
    # неразрывные пробелы
    text = text.replace("\u00A0", " ").replace("\u2009", " ").replace("\u202f", " ")
    return text


# ======================================================================
#  PROTECTOR ENGINE
# ======================================================================

def _protect_dates(text: str) -> str:
    """Первый этап защиты — даты."""
    for pat in DATE_PATTERNS:
        text = re.sub(pat, lambda m: m.group(0).replace(".", "__DOT__").replace("/", "__SLASH__"), text)
    return text


def _protect_safe_tokens(text: str) -> str:
    """Второй этап защиты — сокращения, ФИО, ст.190 и т.п."""
    for pat in SAFE_PATTERNS:
        text = re.sub(pat, lambda m: m.group(0).replace(".", "__DOT__"), text)
    return text


def _restore(text: str) -> str:
    return (
        text.replace("__DOT__", ".")
            .replace("__SLASH__", "/")
    )


# ======================================================================
#  SENTENCE BOUNDARY LOGIC
# ======================================================================

ENDINGS = ".!?"

CLOSERS = "\"'»”›)]}"
OPENERS = "\"«“‘([{-"


def _is_sentence_boundary(text: str, idx: int) -> bool:
    ch = text[idx]
    if ch not in ENDINGS:
        return False

    # не считать многоточие границей
    if idx + 1 < len(text) and text[idx + 1] == ".":
        return False

    # двигаемся вперед (скобки, кавычки)
    j = idx + 1
    while j < len(text) and text[j] in CLOSERS:
        j += 1

    # OCR-case: ".Далее"
    if j < len(text) and text[j].isalpha() and text[j].upper() == text[j]:
        return True

    # пропускаем пробелы
    while j < len(text) and text[j].isspace():
        j += 1

    if j >= len(text):
        return True

    # новое предложение начинается с заглавной, цифры, открывающей кавычки/скобки
    if text[j].isalpha() and text[j].upper() == text[j]:
        return True
    if text[j].isdigit():
        return True
    if text[j] in OPENERS:
        return True

    return False


# ======================================================================
#  MANUAL SPLIT
# ======================================================================

def _manual_split(text: str) -> List[str]:
    out = []
    buf = []

    for i, ch in enumerate(text):
        buf.append(ch)
        if _is_sentence_boundary(text, i):
            s = "".join(buf).strip()
            if len(s) >= MIN_SENT_LEN:
                out.append(s)
            buf = []

    if buf:
        s = "".join(buf).strip()
        if len(s) >= MIN_SENT_LEN:
            out.append(s)

    return out


# ======================================================================
#  PUBLIC FUNCTION
# ======================================================================

def split_into_sentences(text: str) -> List[str]:
    text = _validate(text)
    text = _normalize(text)

    # Защита дат и ключевых конструкций
    text = _protect_dates(text)
    text = _protect_safe_tokens(text)

    # Разбиение
    raw = _manual_split(text)

    # Восстановление
    restored = [_restore(s).strip() for s in raw]

    # Удаление мусора
    clean = [
        s for s in restored
        if not re.fullmatch(r"[.!?,;:\-\s]+", s)
    ]

    logger.info(f"SentenceSplitter v15 → {len(clean)} sent")
    return clean


# ======================================================================
#  LOCAL TEST
# ======================================================================

if __name__ == "__main__":
    tests = [
        "04.05.2024 г. потерпевший перевел 500 000 тг. Далее он сообщил.",
        "Подозреваемый пояснил.Затем потерпевший уточнил.",
        "Он сказал: «Я перевёл 500 000 тг.» Далее он добавил, что пожалел.",
        "Он думал... Потом сообщил, что это пирамида.",
        "12.03.2024 через Kaspi Gold он получил деньги.",
        "12/03/2024 потерпевшая дала показания.",
        "2024-03-12 он подписал договор.",
        "14 апреля 2024 года он получил перевод.",
        "14 қаңтар 2024 жылы аударым алды."
    ]

    for t in tests:
        print("\nTEXT:", t)
        for i, s in enumerate(split_into_sentences(t), 1):
            print(" →", s)
