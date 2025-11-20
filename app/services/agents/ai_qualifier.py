# app/services/agents/ai_qualifier.py
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from app.services.llm_client import LLMClient
from app.services.agents.ai_laws import ALL_AFM_LAWS
from app.services.agents.ai_extractor import extract_all, super_pre_filter
from app.services.reranker import LLMReranker
from app.services.validation.verifier import run_full_verification
from app.services.agents import prompts

logger = logging.getLogger(__name__)

# ============================================================
# ⚙️ Глобальные настройки / константы
# ============================================================

MODEL_VERSION = "qualifier-llm-3.0"
MIN_FACT_CONFIDENCE = 0.5
CONTEXT_RADIUS = 60

_llm_client = LLMClient()


# ============================================================
# 🧩 Кастомные ошибки
# ============================================================

class LLMUnavailableError(Exception):
    """Исключение при недоступности или ошибке LLM."""
    pass


# ============================================================
# 🔌 Обёртка над LLMClient (единая точка вызова)
# ============================================================

def _ask_llm(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Вызов LLM через общий клиент.
    Если LLM недоступен / вернул ошибку — поднимаем LLMUnavailableError.
    """
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        content = _llm_client.chat(messages)
    except Exception as e:
        logger.error(f"Ошибка вызова LLM: {e}")
        raise LLMUnavailableError(str(e))

    if not content or (isinstance(content, str) and content.startswith("[LLM ERROR]")):
        raise LLMUnavailableError(content or "Пустой ответ LLM")

    return str(content).strip()


# ============================================================
# 🧮 Регулярки для первичного извлечения
# ============================================================

PERSON_RX = re.compile(
    r"\b([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]\.){1,2}|[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2})\b"
)
DATE_RX = re.compile(
    r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"
)
MONEY_RX = re.compile(
    r"(?:(\d{1,3}(?:\s?\d{3})+|\d+)(?:[.,]\d{1,2})?)\s?(?:тг|тенге|KZT|₸)",
    re.IGNORECASE,
)
ART_RX = re.compile(
    r"(ст\.?|стать[ьяи])\s*([0-9]{1,3}(?:[-–][0-9]+)?)(?:\s*(УК|УПК|ГК)\s*РК)?",
    re.IGNORECASE,
)

SENTENCE_SPLIT_RX = re.compile(r"(?<=[\.\?\!])\s+")


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = SENTENCE_SPLIT_RX.split(text)
    return [p.strip() for p in parts if p.strip()]


# ============================================================
# 🔎 Фильтр чисто процессуального текста / предложений
# ============================================================

def _is_procedural_sentence(t: str) -> bool:
    lt = t.lower()

    blocked = [
        "потерпевший имеет право",
        "подозреваемый имеет право",
        "разъяснены права",
        "ему разъяснены права",
        "ей разъяснены права",
        "разъяснены обязанности",
        "уголовно-процессуальный кодекс",
        "ст. 64 упк рк",
        "ст. 71 упк рк",
        "ст. 73 упк рк",
        "вопрос:", "ответ:",
        "допрос начат", "допрос окончен",
        "следственное действие начато",
        "следственное действие окончено",
        "кабинет №", "служебный кабинет",
        "аудиозапись", "видеозапись",
        "звуко- и (или) видеозапись",
    ]

    tech = [
        "qr-код", "qr код",
        "эцп", "ecp",
        "электронный документ",
        "дата и время подписания",
        "подпись наложена",
    ]

    return any(b in lt for b in blocked) or any(tk in lt for tk in tech)


def _is_fact_sentence(t: str) -> bool:
    """
    Критерий: предложение содержит СУЩЕСТВЕННЫЙ факт (деньги, действия, роль, ущерб и т.п.).
    """
    if not t:
        return False

    lt = t.lower().strip()
    if len(lt) < 15:
        return False

    # процессуалку выбрасываем
    if _is_procedural_sentence(lt):
        return False

    # 1) Упоминание ключевых ролей
    if any(w in lt for w in ["подозреваем", "обвиняем", "потерпевш", "свидетел"]):
        return True

    # 2) Прямые действия с деньгами
    money_actions = [
        "внес", "внесла", "внесены",
        "перевел", "перевела", "перечислил", "перечислила",
        "передал", "передала",
        "отправил", "отправила",
        "пополнил", "пополнила",
        "снял", "сняла", "вывел", "вывела",
        "получил", "получила",
    ]
    if any(w in lt for w in money_actions):
        return True

    # 3) Ущерб / невозврат
    if any(w in lt for w in ["ущерб", "денег не вернули", "деньги пропали", "не вернули деньги"]):
        return True

    # 4) Обман / инвестиции
    if any(w in lt for w in ["обман", "ввел в заблуждение", "вложил", "вложила", "инвестиц", "доход", "вознагражден"]):
        return True

    # 5) Явная сумма
    if MONEY_RX.search(t):
        return True

    # 6) Хронология / события
    if any(w in lt for w in ["произошло", "случилось", "после этого", "в дальнейшем", "в тот же день"]):
        return True

    return False


# ============================================================
# 🔎 Инфраструктура для протоколов допроса
# ============================================================

def _looks_like_interrogation_doc(doc: Dict[str, Any]) -> bool:
    name = (doc.get("filename") or "").lower()
    t0 = (doc.get("text") or "").lower()[:400]

    return (
        any(k in name for k in ["допрос", "опрос", "объясн", "пояснен"])
        or "протокол допроса" in t0
        or "протокол опроса" in t0
    )


def _clean_interrogation_text(raw: str) -> str:
    """
    Убирает шапку, предупреждения и вопросы из протокола допроса.
    Оставляет только фактические ответы и повествование.
    """
    lines = raw.splitlines()
    cleaned: List[str] = []
    in_body = False

    for line in lines:
        l = line.strip()
        if not l:
            continue

        low = l.lower()

        # служебное — пропускаем
        if any(k in low for k in [
            "перед началом следственного действия",
            "разъяснены права",
            "предупрежден", "предупреждена",
            "об ответственности по ст.",
            "ему разъяснено", "ей разъяснено",
            "копию протокола получил",
        ]):
            continue

        # убираем вопросы
        if low.startswith("вопрос:") or low.startswith("вопрос №"):
            continue

        # начало тела
        if not in_body and any(k in low for k in ["пояснил", "пояснила", "сообщил", "сообщила", "показал", "показала"]):
            in_body = True

        if in_body:
            cleaned.append(l)

    return "\n".join(cleaned) if cleaned else raw


# ============================================================
# 🔎 Извлечение фактов и базовых сущностей из docs
# ============================================================

def _extract_facts_and_sources(
    docs: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str], List[str], List[str], List[Dict[str, Any]]]:
    """
    FACT-BUILDER:
    • Каждый факт = одно предложение из исходных документов.
    • У каждого факта есть минимум один источник {file_id, page}.
    """
    facts: List[Dict[str, Any]] = []
    persons: List[str] = []
    dates: List[str] = []
    amounts: List[str] = []
    sources: List[Dict[str, Any]] = []
    fact_id = 1

    for d in docs:
        text = (d.get("text") or "").strip()
        file_id = d.get("file_id")
        page = d.get("page")

        if not text:
            continue

        # очистка протокола допроса
        if _looks_like_interrogation_doc(d):
            text = _clean_interrogation_text(text)
            if not text.strip():
                continue

        src = {"file_id": file_id, "page": page}
        if file_id:
            sources.append(src)

        # сущности на уровне куска
        for m in PERSON_RX.finditer(text):
            p = m.group(1)
            if p and len(p) > 2 and not any(x in p for x in ["АО", "ТОО", "ИП", "ООО"]):
                if p not in persons:
                    persons.append(p)

        for m in DATE_RX.finditer(text):
            dt = m.group(1)
            if dt not in dates:
                dates.append(dt)

        for m in MONEY_RX.finditer(text):
            amt = m.group(0)
            if amt not in amounts:
                amounts.append(amt)

        # предложения → факты
        for sent in _split_sentences(text):
            sent_clean = sent.strip()
            if not sent_clean:
                continue
            if not _is_fact_sentence(sent_clean):
                continue

            low = sent_clean.lower()
            if "подозреваем" in low and "признать" in low:
                fact_type = "status"
            else:
                fact_type = "event"

            facts.append(
                {
                    "fact_id": f"f{fact_id}",
                    "type": fact_type,
                    "text": sent_clean[:600],
                    "confidence": _conf_from_signal(sent_clean),
                    "sources": [src] if file_id else [],
                }
            )
            fact_id += 1

    # fallback: если нет фактов, но есть сущности
    if not facts and (persons or dates or amounts):
        base_parts = []
        if persons:
            base_parts.append(f"Участники: {', '.join(persons[:5])}")
        if dates:
            base_parts.append(f"Даты: {', '.join(dates[:5])}")
        if amounts:
            base_parts.append(f"Суммы: {', '.join(amounts[:5])}")

        if base_parts:
            facts.append(
                {
                    "fact_id": f"f{fact_id}",
                    "type": "context",
                    "text": "; ".join(base_parts),
                    "confidence": 0.55,
                    "sources": [sources[0]] if sources else [],
                }
            )

    uniq_sources = _dedup_sources(sources)
    return facts, persons, dates, amounts, uniq_sources


# ============================================================
# 🧠 Обогащение фактов ролями / действиями
# ============================================================

def enrich_facts_with_roles(facts: List[dict]) -> List[dict]:
    ROLE_HINTS = {
        "подозреваем": "подозреваемый",
        "обвиня": "обвиняемый",
        "свидетел": "свидетель",
        "потерпевш": "потерпевший",
        "соучаст": "соучастник",
        "организатор": "организатор",
    }
    ACTION_HINTS = [
        "перевёл", "передал", "получил", "принял", "предложил",
        "обманул", "ввел в заблуждение", "совершил", "присвоил", "вымогал",
        "организовал", "заключил договор", "получил доступ", "снял деньги",
    ]

    for f in facts:
        txt = (f.get("text") or "").lower()
        f["role"] = next((r for k, r in ROLE_HINTS.items() if k in txt), "неопределено")
        f["action"] = next((a for a in ACTION_HINTS if a in txt), None)
        f["time"] = next(
            (d for d in re.findall(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}", txt)), None
        )
    return facts


# ============================================================
# 🧱 Проверка наличия данных о подозреваемом
# ============================================================

def validate_facts_completeness(docs: List[Dict[str, Any]]) -> None:
    """
    Быстрая проверка: есть ли ВООБЩЕ документы и упоминание подозреваемого.
    Вызывается из API-роутера ДО запуска квалификатора.
    """
    if not docs:
        raise HTTPException(
            status_code=400,
            detail="❌ Не найдены документы для анализа. Проверьте, загружены ли файлы по делу.",
        )

    has_suspect = any("подозреваем" in (d.get("text") or "").lower() for d in docs)
    if not has_suspect:
        raise HTTPException(
            status_code=404,
            detail=(
                "❌ В текстах не обнаружены сведения о подозреваемом. "
                "Требуется проверить OCR и полноту загруженных материалов."
            ),
        )


# ============================================================
# ✅ Проверка полноты по ст. 204 УПК РК
# ============================================================

def _check_204_completeness(
    facts: List[Dict[str, Any]],
    persons: List[str],
    dates: List[str],
    amounts: List[str],
    roles: Optional[Dict[str, Any]] = None,
    events: Optional[List[Dict[str, Any]]] = None,
    legal_facts: Optional[Dict[str, Any]] = None,
    timeline: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    roles = roles or {}
    events = events or []
    legal_facts = legal_facts or {}

    def present(x: Any) -> bool:
        return bool(x)

    checklist = [
        {
            "item": "Установлена личность подозреваемого",
            "present": present(roles.get("suspect")),
        },
        {
            "item": "Определена роль подозреваемого",
            "present": present(roles.get("suspect_role")),
        },
        {
            "item": "Есть описание событий",
            "present": present(events),
        },
        {
            "item": "Есть даты событий",
            "present": present(dates),
        },
        {
            "item": "Есть финансовые сведения",
            "present": present(amounts),
        },
        {
            "item": "Выделены юридически значимые признаки",
            "present": present(legal_facts),
        },
        {
            "item": "Есть фактические данные (выдержки из документов)",
            "present": present(facts),
        },
    ]

    missing = [x["item"] for x in checklist if not x["present"]]

    return {
        "article": "204 УПК РК",
        "checklist": checklist,
        "missing": missing,
        "enough_for_draft": len(missing) <= 3,
    }


# ============================================================
# ⚖️ Извлечение статей из текстов
# ============================================================

def _extract_articles(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    arts: List[Dict[str, Any]] = []
    for d in docs:
        text = d.get("text") or ""
        file_id = d.get("file_id")
        page = d.get("page")
        for m in ART_RX.finditer(text):
            art_num = m.group(2)
            code = (m.group(3) or "УК/УПК/ГК?").upper()
            arts.append(
                {
                    "code": (code.replace(" ", "") + " РК") if "РК" not in code else code,
                    "article": art_num,
                    "context": _context_snippet(text, m.start(), m.end()),
                    "source": {"file_id": file_id, "page": page},
                }
            )
    return _dedup_articles(arts)


def _resolve_law_context(article_num: str) -> str:
    data = ALL_AFM_LAWS.get(article_num)
    if not data:
        return f"Статья {article_num}: [описание отсутствует в базе AFM]"

    code = data.get("code", "УК/УПК РК")
    name = data.get("name", "[название отсутствует]")
    official_text = data.get("official_text") or data.get("text") or "[текст отсутствует]"
    category = data.get("category", "прочее")

    return f"{code} ст.{article_num} — {name}. {official_text} ({category})"


# ============================================================
# 🧾 Построение раздела «УСТАНОВИЛ» (базовый, детерминированный)
# ============================================================

def _build_ustanovil_base(
    facts: List[dict],
    completeness: Dict[str, Any],
    suspect: Optional[str] = None,
    suspect_role: Optional[str] = None,
) -> str:
    """
    Базовая детерминированная версия «УСТАНОВИЛ»:
    - просто нумерованный список фактов
    - без переформулирования
    - с [file_id:page] в конце каждого факта
    Используется как fallback и вход для LLM.
    """
    if not facts and not suspect:
        return "Существенных фактов не обнаружено. Требуется дополнительная проверка."

    lines: List[str] = ["УСТАНОВИЛ:"]

    # Блок о подозреваемом (если есть)
    if suspect:
        line = f"Из материалов дела усматривается, что {suspect}"
        if suspect_role:
            line += f", выполняя роль {suspect_role},"
        else:
            line += ","
        line += " фигурирует в качестве лица, в отношении которого проводится досудебное расследование."
        lines.append(line)
        lines.append("")

    # Факты
    for i, f in enumerate(facts, 1):
        src_str = _src_str(f.get("sources"))
        conf = f.get("confidence", 0.5)
        suffix = "" if conf >= 0.75 else " [⚠️ низкая уверенность]"
        lines.append(f"{i}. {f['text']} {src_str}{suffix}")

    # Недостающие элементы
    missing = completeness.get("missing") or []
    if missing:
        lines.append("")
        lines.append("Недостающие элементы для полной квалификации по ст. 204 УПК РК:")
        for m in missing:
            lines.append(f"• {m}")

    return "\n".join(lines)


# ============================================================
# 🧾 Fallback-постановление (без LLM)
# ============================================================

def _build_postanovlenie_simple(
    city: str,
    date_str: str,
    investigator_line: str,
    case_id: Optional[str],
    ustanovil_text: str,
    mentioned_articles: List[Dict[str, Any]],
    completeness: Dict[str, Any],
    investigator_fio: str = "",
    intro_context: str = "",
) -> str:
    rus_date = _rus_date(date_str)

    # список упоминаний статей УК
    if mentioned_articles:
        arts_filtered = [a for a in mentioned_articles if "УК" in a.get("code", "")]
        if arts_filtered:
            arts = sorted({f"{a.get('code', '')} ст.{a.get('article', '?')}" for a in arts_filtered})
            arts_line = "Упоминания статей УК: " + "; ".join(arts)
        else:
            arts_line = "Упоминаний статей УК не выявлено."
    else:
        arts_line = "Упоминаний статей нет."

    # решение
    if completeness.get("enough_for_draft"):
        decision = "Квалифицировать деяние подозреваемого по соответствующим статьям УК Республики Казахстан."
    else:
        decision = "Окончательную квалификацию определить после получения недостающих материалов."

    intro_block = ""
    if intro_context:
        intro_block = intro_context.strip() + "\n\n"

    return f"""ПОСТАНОВЛЕНИЕ
о квалификации деяния подозреваемого

{city}, {rus_date}

Материалы дела № {case_id}
{arts_line}

{intro_block}{ustanovil_text}

ПОСТАНОВИЛ:
{decision}

Подпись:
Следователь: {investigator_line}
ФИО: {investigator_fio}
______________________
Дата: {rus_date}

Черновик сформирован автоматически; окончательное решение принимает следователь после проверки и утверждения прокурором.
""".strip()


# ============================================================
# 🎯 Фильтры юридически значимых фактов
# ============================================================

def _legal_fact_filter(fact_text: str) -> bool:
    if not fact_text:
        return False

    t = fact_text.lower().strip()

    rights_noise = [
        "ст. 67 упк рк",
        "статья 67 упк рк",
        "потерпевший имеет право",
        "подозреваемый имеет право",
        "права и обязанности потерпевшего",
        "права и обязанности подозреваемого",
        "он может быть подвергнут приводу",
        "наложено денежное взыскание",
        "процессуальные права",
    ]
    if any(w in t for w in rights_noise):
        return False

    blocked_pdf = [
        "qr-код", "qr код", "хеш", "хэш", "hash",
        "ecp", "эцп", "электронный документ",
        "данные эцп", "код документа", "время подписания",
        "датой и временем подписания", "электронный pdf",
    ]
    if any(w in t for w in blocked_pdf):
        return False

    if "разъясн" in t and "прав" in t and "показан" not in t and "пояснил" not in t:
        return False

    if "явиться по вызову" in t or "не разглашать сведения" in t:
        return False

    if "видеокамера" in t or "видеозапись" in t or "аудиозапись" in t:
        return False

    intro_markers = [
        "проводится досудебное расследование",
        "рассмотрев материалы досудебного расследования",
        "рассмотрев материалы уголовного дела",
        "материалы досудебного расследования №",
        "материалы уголовного дела №",
        "материалы дела №",
        "руководитель сог",
        "руководитель су дер",
        "руководитель следственной-оперативной группы",
    ]
    if any(m in t for m in intro_markers):
        return False

    plan_markers = [
        "санкционировать производство обысковых мероприятий",
        "санкционировать производство обыска",
        "направить в территориальные департаменты",
        "проанализировать приобретение имущества",
        "истребовать справки и декларации о доходах",
        "поручить провести",
        "провести оперативно-розыскные мероприятия",
    ]
    if any(m in t for m in plan_markers):
        return False

    if len(t) < 15:
        return False

    # если есть признаки финансовых действий / ущерба
    crime_keywords = [
        "пирамид", "вовлек", "привлек", "внес", "внесла",
        "вложил", "вложила", "перевел", "перевела", "перечислил",
        "денежн", "деньги", "средства", "ущерб", "баланс", "usdt",
        "приложени", "платформ", "инвест", "доход", "вознаграждение",
    ]
    if any(w in t for w in crime_keywords):
        return True

    action_words = ["соверш", "организ", "руковод", "получил", "получила", "присво", "обманул"]
    if any(w in t for w in action_words):
        return True

    return True


def _hard_fact_clean(fact_text: str) -> bool:
    if not fact_text:
        return False

    t = fact_text.lower().strip()

    noise = [
        "я лично ничего не помню",
        "я не знаю", "мы думали", "как-то", "вроде",
        "мама сказала", "сосед рассказал",
    ]
    if any(p in t for p in noise):
        return False

    procedural = [
        "упк рк", "уголовно-процессуальный кодекс",
        "имеет право", "обязан", "обязана",
        "разъяснены права", "права и обязанности",
        "предупрежден об ответственности", "предупреждена об ответственности",
    ]
    if any(p in t for p in procedural) and "пояснил" not in t and "пояснила" not in t:
        return False

    tech = ["qr-код", "qr код", "ecp", "эцп", "pdf", "скан-копия"]
    if any(k in t for k in tech):
        return False

    return True


# ============================================================
# 🤖 Авто-квалификация по ключевым словам
# ============================================================

def _auto_qualify(
    facts: List[Dict[str, Any]],
    roles: Dict[str, Any],
    events: List[Dict[str, Any]],
    legal_facts: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str], str]:
    text_all = " ".join(f.get("text", "").lower() for f in facts)

    pyramid_keywords = [
        "пирамида", "вложил", "привлек", "завлекал",
        "вовлекал", "вступил", "схема", "доход за счет новых участников",
        "перераспределение", "инвест проект без актива",
    ]
    if any(k in text_all for k in pyramid_keywords):
        return (
            "217",
            "1",
            "Обнаружены признаки финансовой пирамиды (привлечение средств, перераспределение вкладов).",
        )

    fraud_keywords = [
        "обман", "заблуждени", "мошеннич", "присвоил",
        "заведомо ложные", "незаконно получил",
    ]
    if any(k in text_all for k in fraud_keywords):
        return (
            "190",
            "2",
            "Обнаружены признаки мошенничества (обман, введение в заблуждение, присвоение).",
        )

    business_keywords = [
        "без лицензии", "незаконная предпринимательская", "нелегальная деятельность",
        "оказание услуг без регистрации", "без удостоверения", "не зарегистрирован",
    ]
    if any(k in text_all for k in business_keywords):
        return (
            "214",
            "1",
            "Обнаружены признаки незаконной предпринимательской деятельности.",
        )

    laundering_keywords = [
        "легализ", "отмывал", "скрывал происхождение", "движение денежных средств",
    ]
    if any(k in text_all for k in laundering_keywords):
        return (
            "218",
            "1",
            "Обнаружены признаки легализации доходов (отмывание денег).",
        )

    return None, None, "Недостаточно данных для автоматической квалификации."


def classify_crime(facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    text_blob = " ".join(f.get("text", "").lower() for f in facts)

    flags = {
        "190": any(kw in text_blob for kw in [
            "обман", "заблуждени", "не вернул", "получил деньги",
            "не выполнил обязательств", "ущерб", "ввел в заблужд",
        ]),
        "217": any(kw in text_blob for kw in [
            "привлек", "вовлек", "обещал доход", "вознаграждение",
            "участник", "структур", "доход за счет других",
            "массовое привлечение", "высокая доходност",
        ]),
        "385": any(kw in text_blob for kw in [
            "без лицензии", "без регистрации", "незаконн",
            "предпринимательская деятельность", "прибыль без регистрации",
        ]),
        "189": any(kw in text_blob for kw in [
            "доверен", "распорядился чужим", "присвоил",
            "растрата", "имущество было передано",
        ]),
        "218": any(kw in text_blob for kw in [
            "скрыть происхождение", "обналич", "перевел между счетами",
            "легализ", "маскиров",
        ]),
    }

    primary = None
    if flags["217"]:
        primary = "217"
    elif flags["190"]:
        primary = "190"
    elif flags["189"]:
        primary = "189"
    elif flags["218"]:
        primary = "218"
    elif flags["385"]:
        primary = "385"

    secondary = [art for art, ok in flags.items() if ok and art != primary]

    return {"primary": primary, "secondary": secondary}


# ============================================================
# 🧠 Процессуальный контекст (до «УСТАНОВИЛ»)
# ============================================================

def _extract_intro_context(docs: List[Dict[str, Any]]) -> str:
    intro_sentences: List[str] = []

    for d in docs:
        text = d.get("text") or ""
        if not text.strip():
            continue

        for sent in _split_sentences(text):
            lt = sent.lower()

            ban = ["постановление", "о признании", "потерпевшим"]
            if any(b in lt for b in ban):
                continue

            if "проводится досудебное расследование" in lt:
                intro_sentences.append(sent.strip())
                continue

            if "рассмотрев материалы досудебного расследования" in lt or \
               "рассмотрев материалы уголовного дела" in lt or \
               "рассмотрев материалы дела" in lt:
                intro_sentences.append(sent.strip())
                continue

            if "материалы досудебного расследования №" in lt or \
               "материалы уголовного дела №" in lt or \
               "материалы дела №" in lt:
                intro_sentences.append(sent.strip())
                continue

            if "руководитель сог" in lt or \
               "руководитель су дер" in lt or \
               "руководитель следственной оперативной группы" in lt:
                intro_sentences.append(sent.strip())
                continue

    seen = set()
    uniq = []
    for s in intro_sentences:
        if s not in seen:
            seen.add(s)
            uniq.append(s)

    return "\n".join(uniq[:2])


# ============================================================
# 🧠 Основная функция квалификации
# ============================================================

def qualify_documents(
    case_id: Optional[str],
    docs: List[Dict[str, Any]],
    city: str = "г. Павлодар",
    date_str: Optional[str] = None,
    investigator_line: str = "Следователь по особо важным делам",
    investigator_fio: str = "",
) -> Dict[str, Any]:
    logger.info(f"▶️ Начало квалификации: case_id={case_id}, документов={len(docs)}")

    if not docs:
        logger.warning("⚠️ Нет документов для анализа")
        return _empty_result(case_id or "", "Документы для анализа отсутствуют")

    if not date_str:
        date_str = datetime.now().strftime("%d.%m.%Y")

    # 0️⃣ Reranker PRO (отдельный сервис)
    try:
        reranker = LLMReranker()
        QUERY = (
            "факты преступления: переводы денег, вложения, вывод средств, "
            "инвестиции, обман, действия подозреваемого, события, суммы, даты"
        )

        docs = reranker.rerank(
            query=QUERY,
            items=docs,
            top_k=200,
        )
        logger.info(f"📊 Reranker PRO: выбрано TOP={len(docs)} документов")
    except Exception as e:
        logger.error(f"❌ Ошибка Reranker: {e}")

    # 1️⃣ SUPER PRE-FILTER + чистый текст
    for d in docs:
        raw = (d.get("text") or "").strip()
        cleaned_sentences = super_pre_filter(raw)
        d["clean_sentences"] = cleaned_sentences
        if cleaned_sentences:
            d["text"] = " ".join(cleaned_sentences)
        else:
            d["text"] = raw

    # 2️⃣ Извлечение фактов / сущностей
    try:
        facts, persons, dates, amounts, sources = _extract_facts_and_sources(docs)
    except Exception as e:
        logger.error(f"_extract_facts_and_sources error: {e}")
        return _empty_result(case_id or "", f"Ошибка извлечения фактов: {e}")

    facts = enrich_facts_with_roles(facts)

    # фильтрация мусора
    raw_count = len(facts)
    facts = [
        f for f in facts
        if _legal_fact_filter(f.get("text", "")) and _hard_fact_clean(f.get("text", ""))
    ]
    facts = [f for f in facts if len((f.get("text") or "").split()) >= 3]
    logger.info(f"ФИЛЬТР ФАКТОВ: было={raw_count}, после фильтра={len(facts)}")

    # 3️⃣ Классификация преступления
    crime_class = classify_crime(facts)
    primary_article = crime_class["primary"]
    secondary_articles = crime_class["secondary"]
    logger.info(f"АВТО-КВАЛИФИКАЦИЯ: primary={primary_article}, secondary={secondary_articles}")

    # 4️⃣ EXTRACTOR 3.0
    extracted = extract_all(facts, persons, dates, amounts)
    roles = extracted.get("roles", {}) or {}
    events = extracted.get("events", []) or []
    timeline = extracted.get("timeline", []) or []
    legal_facts = extracted.get("legal_facts", {}) or {}
    crime_flow = extracted.get("crime_flow", []) or []
    crime_type = extracted.get("crime_type")

    suspects_list = extracted.get("suspects") or roles.get("suspect", []) or []
    suspect = extracted.get("primary_suspect") or (suspects_list[0] if suspects_list else None)
    suspect_role = roles.get("suspect_role")

    logger.info(
        f"[EXTRACTOR] suspect={suspect}, events={len(events)}, crime_type={crime_type}, "
        f"timeline={len(timeline)}, flow={len(crime_flow)}"
    )

    # усиление legal_facts hint'ами
    legal_facts.update(
        {
            "crime_type": crime_type,
            "primary_article_hint": primary_article,
            "secondary_articles_hint": secondary_articles,
            "has_flow": bool(crime_flow),
        }
    )

    # 5️⃣ Дополнительная авто-квалификация
    auto_article, auto_part, auto_reason = _auto_qualify(
        facts=facts,
        roles=roles,
        events=events,
        legal_facts=legal_facts,
    )
    logger.info(f"Авто-квалификация (auto_qualify): статья={auto_article}, часть={auto_part} — {auto_reason}")

    # 6️⃣ Полнота по ст. 204 УПК
    completeness = _check_204_completeness(
        facts=facts,
        persons=persons,
        dates=dates,
        amounts=amounts,
        roles=roles if suspect else {},
        events=events,
        legal_facts=legal_facts,
        timeline=timeline,
    )

    # 7️⃣ Упоминания статей + контекст АФМ
    mentioned_articles = _extract_articles(docs)
    logger.info(f"Упоминаний статей: {len(mentioned_articles)}")

    law_contexts: List[str] = []
    for art in mentioned_articles:
        num = art.get("article")
        if num and num in ALL_AFM_LAWS:
            law_contexts.append(_resolve_law_context(num))
    law_context_text = "\n".join(law_contexts[:5]) if law_contexts else ""

    # 8️⃣ Базовый «УСТАНОВИЛ» (с ссылками)
    ustanovil_base = _build_ustanovil_base(
        facts=facts,
        completeness=completeness,
        suspect=suspect,
        suspect_role=suspect_role,
    )

    # 9️⃣ Попытка улучшить «УСТАНОВИЛ» через LLM (юридический стиль МВД)
    ustanovil_llm = ustanovil_base
    try:
        if facts:
            fact_lines: List[str] = []
            for idx, f in enumerate(facts, 1):
                fact_lines.append(f"{idx}. {f['text']} {_src_str(f.get('sources'))}")

            missing_text = ", ".join(completeness.get("missing", [])) or "нет"

            suspect_block = ""
            if suspects_list:
                if len(suspects_list) == 1:
                    suspect_block = f"ПОДОЗРЕВАЕМЫЙ: {suspects_list[0]}"
                else:
                    suspect_block = "ПОДОЗРЕВАЕМЫЕ:\n" + "\n".join(f"- {s}" for s in suspects_list)
            else:
                suspect_block = "ПОДОЗРЕВАЕМЫЕ: не установлены"

            user_prompt = f"""
Ниже передан список ОРИГИНАЛЬНЫХ фактов, извлечённых из материалов
уголовного дела (рапорты, постановления, протоколы допроса и т.п.).

Каждый факт:
• является готовым предложением из документа;
• содержит ссылку на источник [file_id:page];
• НЕ может быть изменён, сокращён или искажён по смыслу.

{suspect_block}

СПИСОК ФАКТОВ:
{chr(10).join(fact_lines)}

Недостающие элементы по ст. 204 УПК РК: {missing_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ТВОЯ ЗАДАЧА — СФОРМИРОВАТЬ раздел «УСТАНОВИЛ»
в стиле постановления следователя МВД.

ТРЕБОВАНИЯ:
• Строгий официальный стиль.
• Разрешено переформулировать текст предложений,
  НО без добавления новых фактов, сумм, дат, участников и обстоятельств.
• Запрещено придумывать действия, эпизоды, последствия, которые отсутствуют в фактах.
• Запрещено ссылаться на статьи УК/УПК, которых нет в фактах и law_context.
• Ссылки [file_id:page] можно сохранять в конце соответствующих предложений.

Структура вывода (СТРОГО):

УСТАНОВИЛ:
<связный юридический текст, построенный ТОЛЬКО из приведённых фактов, без новых сведений>

Никаких иных заголовков, комментариев, служебных пометок.
"""

            system_prompt = """
Ты — специализированный юридический модуль «AI_Qualifier» для органов финансового мониторинга.
Твоя задача — на основе готовых фактов сформировать раздел «УСТАНОВИЛ» в стиле постановления следователя.

СТРОГО ЗАПРЕЩЕНО:
• добавлять новые факты, суммы, даты, ФИО, организации, эпизоды;
• менять причинно-следственные связи, если они прямо не следуют из фактов;
• придумывать мотивы, умысел, квалификацию деяния;
• ссылаться на источники, которых нет в фактах.

РАЗРЕШЕНО:
• переформулировать предложения, сохраняю смысл;
• объединять близкие по смыслу факты в абзацы;
• менять порядок, если это делает изложение логичным;
• оставлять ссылки [file_id:page] в конце соответствующих предложений.

Если фактов явно недостаточно — сформируй короткий формальный текст
о том, что существенных фактических данных недостаточно для окончательных выводов.
"""

            ustanovil_llm = _ask_llm(user_prompt, system_prompt)
            logger.info("✅ Раздел «УСТАНОВИЛ» сформирован LLM.")
    except LLMUnavailableError as e:
        logger.warning(f"LLM недоступен для раздела «УСТАНОВИЛ»: {e}")
        ustanovil_llm = ustanovil_base

    # 10️⃣ Генерация «ПОСТАНОВИЛ»
    safe_article = primary_article or auto_article or "[Требует уточнения]"

    safe_primary = safe_article or "не определена"
    law_data = ALL_AFM_LAWS.get(safe_primary, {})

    post_prompt = prompts.P_POST.format(
        ustanovil_text = ustanovil_llm,
        primary_article = safe_primary,
        secondary_articles = ", ".join(secondary_articles) if secondary_articles else "нет",
        law_text = law_data.get("text", "Текст статьи не найден"),
        law_commentary = law_data.get("commentary", ""),
    )


    system_for_post = (
        "Ты — узкоспециализированный модуль «AI_Qualifier_Post». "
        "Твоя единственная задача — сформировать юридически грамотный раздел «ПОСТАНОВИЛ» "
        "исключительно на основе текста раздела «УСТАНОВИЛ».\n\n"
        "СТРОГО ЗАПРЕЩЕНО:\n"
        "- добавлять новые факты, события, суммы, даты, участников;\n"
        "- придумывать обстоятельства, которых нет в разделе «УСТАНОВИЛ»;\n"
        "- ссылаться на органы и структуры, которых нет в тексте;\n"
        "- давать советы следователю или оценочные суждения.\n\n"
        "Разрешено только кратко сформулировать решение: "
        "предварительная квалификация, необходимость дополнительных материалов и т.п.\n"
        "Структура итогового текста: УСТАНОВИЛ → ПОСТАНОВИЛ → подпись (без шапок, гербов и лишних блоков)."
    )

    intro_context = _extract_intro_context(docs)

    try:
        full_user_prompt = f"""
Автоматически выявленная статья УК РК (классификатор): {primary_article or "не определена"}.
Автоматически выявленная статья УК РК (анализ фактов): {auto_article or "не определена"} ч.{auto_part or "-"}.
Причина: {auto_reason}.

Предварительный тип преступления (без LLM): {crime_type or "неопределено"}.

Материалы дела № {case_id}.
Место вынесения: {city}.
Дата: {_rus_date(date_str)}.

Процессуальный контекст (его нужно ДОСЛОВНО вставить перед разделом «УСТАНОВИЛ», если он не пустой):
{intro_context or "[нет процессуального контекста]"}

Вспомогательный юридический контекст (если есть):
{law_context_text or "[нет дополнительных формулировок законов]"}

Основывайся исключительно на разделе «УСТАНОВИЛ» ниже и на блоке процессуального контекста выше.

{post_prompt}

Статья для основной квалификации (если это подтверждается фактами): {safe_article}.
Если данных недостаточно — в разделе «ПОСТАНОВИЛ» отрази необходимость получения
дополнительных доказательств перед окончательной квалификацией.
"""
        final_postanovlenie_raw = _ask_llm(
            prompt=full_user_prompt,
            system_prompt=system_for_post,
        )
        logger.info("✅ Постановление сгенерировано через LLM.")
    except LLMUnavailableError as e:
        logger.warning(f"LLM недоступен для Постановления: {e}")
        final_postanovlenie_raw = _build_postanovlenie_simple(
            city=city,
            date_str=date_str,
            investigator_line=investigator_line,
            case_id=case_id,
            ustanovil_text=ustanovil_llm,
            mentioned_articles=mentioned_articles,
            completeness=completeness,
            investigator_fio=investigator_fio,
            intro_context=intro_context,
        )

    # 11️⃣ Страховка структуры
    lower_body = final_postanovlenie_raw.lower()
    if "установил" not in lower_body or "постановил" not in lower_body:
        logger.warning("⚠️ LLM отклонился от структуры, применяю fallback-шаблон Постановления.")
        final_postanovlenie_raw = _build_postanovlenie_simple(
            city=city,
            date_str=date_str,
            investigator_line=investigator_line,
            case_id=case_id,
            ustanovil_text=ustanovil_llm,
            mentioned_articles=mentioned_articles,
            completeness=completeness,
            investigator_fio=investigator_fio,
            intro_context=intro_context,
        )

    # 12️⃣ Оценка уверенности
    overall_conf = _overall_confidence(facts, completeness)

    # 13️⃣ Базовый result (С ССЫЛКАМИ — для верификатора)
    result: Dict[str, Any] = {
        "generation_id": str(uuid.uuid4()),
        "model_version": MODEL_VERSION,
        "case_id": case_id,
        "facts": facts,
        "persons": persons,
        "dates": dates,
        "amounts": amounts,
        "mentioned_articles": mentioned_articles,
        "roles": roles,
        "events": events,
        "timeline": timeline,
        "legal_facts": legal_facts,
        "completeness_204": completeness,
        "established_text": ustanovil_llm.strip(),          # ← с [file_id:page]
        "final_postanovlenie": final_postanovlenie_raw.strip(),  # ← тоже может содержать ссылки
        "sources": sources,
        "confidence": round(overall_conf, 3),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "investigator_fio": investigator_fio,
        "investigator_line": investigator_line,
        "auto_article": auto_article,
        "auto_part": auto_part,
        "auto_reason": auto_reason,
        "auto_classification": crime_class,
        "suspect": suspect,
        "suspect_role": suspect_role,
        "crime_flow": crime_flow,
        "crime_type": crime_type,
        "warnings": [],
    }

    # 14️⃣ Anti-hallucination верификация (работает по тексту С ССЫЛКАМИ)
    try:
        verification = run_full_verification(result)
        if not isinstance(verification, dict):
            raise ValueError("Некорректный формат результата верификации")

        result["verification"] = verification
        result["verdict"] = (
            "OK"
            if verification.get("overall_ok")
            else verification.get("texts", {}).get("verdict", "UNVERIFIED")
        )

        if not verification.get("overall_ok"):
            result["warnings"].append("⚠️ Верификация выявила потенциальные неточности")
    except Exception as e:
        logger.error(f"Ошибка верификации: {e}")
        result["verification"] = {"error": str(e)}
        result["verdict"] = "VERIFICATION_FAILED"
        result["warnings"].append(f"Ошибка верификации: {str(e)}")

    # 15️⃣ Удаляем ссылки из текстов перед отдачей наружу (Вариант B)
    result["established_text"] = _remove_sources(result.get("established_text", ""))
    result["final_postanovlenie"] = _remove_sources(result.get("final_postanovlenie", ""))

    logger.info(
        f"✅ Квалификация завершена. verdict={result.get('verdict')}, "
        f"conf={result.get('confidence'):.2f}, suspect={suspect}"
    )
    return result


# ============================================================
# 🔹 Fallback-результат при ошибках
# ============================================================

def _empty_result(case_id: str, msg: str,
                  investigator_fio: str = "",
                  investigator_line: str = "") -> Dict[str, Any]:
    return {
        "generation_id": None,
        "model_version": MODEL_VERSION,
        "case_id": case_id,
        "established_text": "",
        "final_postanovленie": f"[ОШИБКА]: {msg}",
        "facts": [],
        "persons": [],
        "dates": [],
        "amounts": [],
        "mentioned_articles": [],
        "roles": {},
        "events": [],
        "timeline": [],
        "legal_facts": {},
        "completeness_204": {},
        "sources": [],
        "confidence": 0.0,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "investigator_fio": investigator_fio,
        "investigator_line": investigator_line,
        "warnings": [msg],
        "verification": {"overall_ok": False},
        "verdict": "ERROR",
        "suspect": None,
        "suspect_role": None,
        "crime_flow": [],
        "crime_type": None,
    }


# ============================================================
# 🧰 Вспомогательные утилиты
# ============================================================

def _remove_sources(text: str) -> str:
    """Удаляет ссылки вида [uuid:page] из текста."""
    return re.sub(r"\[[0-9a-fA-F\-]{36}:\d+\]", "", text)


def _context_snippet(text: str, start: int, end: int, radius: int = CONTEXT_RADIUS) -> str:
    a, b = max(0, start - radius), min(len(text), end + radius)
    return text[a:b].replace("\n", " ").strip()


def _src_str(sources: Optional[List[Dict[str, Any]]]) -> str:
    if not sources:
        return "[источник: не указан]"
    show = [
        f"[{s.get('file_id', '?')}:{s.get('page', '-')}]"
        for s in sources[:3]
    ]
    if len(sources) > 3:
        show.append(f"(и ещё {len(sources) - 3})")
    return " ".join(show)


def _conf_from_signal(sentence: str) -> float:
    score = MIN_FACT_CONFIDENCE
    if DATE_RX.search(sentence):
        score += 0.15
    if MONEY_RX.search(sentence):
        score += 0.15
    if PERSON_RX.search(sentence):
        score += 0.1
    return min(score, 0.95)


def _dedup_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for s in sources:
        key = (s.get("file_id"), s.get("page"))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _dedup_articles(arts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for a in arts:
        key = (
            a["code"],
            a["article"],
            a["source"].get("file_id"),
            a["source"].get("page"),
        )
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


def _overall_confidence(facts: List[dict], completeness: Dict[str, Any]) -> float:
    if not facts:
        return 0.4
    avg = sum(f.get("confidence", 0.5) for f in facts) / max(1, len(facts))
    miss_penalty = 0.05 * len(completeness.get("missing", []))
    return max(0.1, min(0.98, avg - miss_penalty))


def _rus_date(d: str) -> str:
    """
    Понимает 'DD.MM.YYYY' и ISO 'YYYY-MM-DD'.
    Если формат неизвестен — возвращает исходную строку.
    """
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
            dt = datetime.fromisoformat(d)
        else:
            dt = datetime.strptime(d, "%d.%m.%Y")
    except Exception:
        return d

    months = [
        "января", "февраля", "марта", "апреля", "мая", "июня",
        "июля", "августа", "сентября", "октября", "ноября", "декабря",
    ]
    return f"{dt.day} {months[dt.month - 1]} {dt.year} года"
