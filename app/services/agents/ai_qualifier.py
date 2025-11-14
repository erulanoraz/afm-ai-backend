# app/services/agents/ai_qualifier.py
from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from app.utils.config import settings
from app.services.validation.verifier import run_full_verification
from app.services.agents.ai_laws import ALL_AFM_LAWS
from app.services.agents.ai_extractor import extract_all
from app.services.llm_client import LLMClient
from app.services.agents import prompts

logger = logging.getLogger(__name__)

# ============================================================
# ⚙️ Глобальные настройки / константы
# ============================================================

MODEL_VERSION = "qualifier-llm-2.1"
MIN_FACT_CONFIDENCE = 0.5
CONTEXT_RADIUS = 60

# LLM-клиент (используем твой общий адаптер)
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

def _ask_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
) -> str:
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

    if not content or isinstance(content, str) and content.startswith("[LLM ERROR]"):
        raise LLMUnavailableError(content or "Пустой ответ LLM")

    return content.strip()


# ============================================================
# 🧮 Регулярки для первичного извлечения
# ============================================================

PERSON_RX = re.compile(
    r"\b([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]\.){1,2}|[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2})\b"
)
DATE_RX = re.compile(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
MONEY_RX = re.compile(
    r"(?:(\d{1,3}(?:\s?\d{3})+|\d+)(?:[.,]\d{1,2})?)\s?(?:тг|тенге|KZT|₸)",
    re.IGNORECASE,
)
ART_RX = re.compile(
    r"(ст\.?|стать[ьяи])\s*([0-9]{1,3}(?:[-–][0-9]+)?)(?:\s*(УК|УПК|ГК)\s*РК)?",
    re.IGNORECASE,
)

EVENT_HINTS = [
    "перевёл", "перевела", "перечислил", "перечислила",
    "получил", "получила", "заключил договор", "заключила договор",
    "подписал", "подписала", "вывод средств", "снятие наличных",
    "хищение", "мошенничество", "незаконное обогащение", "присвоение"
]


# ============================================================
# 🔎 Извлечение фактов и базовых сущностей из docs
# ============================================================

def _extract_facts_and_sources(
    docs: List[Dict[str, Any]]
) -> tuple[list[dict], list[str], list[str], list[str], list[dict]]:
    """
    docs приходит из retrieval.get_file_docs_for_qualifier в формате:
    {
        "file_id": "uuid",
        "page": 1,
        "chunk_id": "uuid",
        "text": "..."
    }
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

        if file_id:
            sources.append({"file_id": file_id, "page": page})

        # 👤 Имена
        for m in PERSON_RX.finditer(text):
            p = m.group(1)
            if len(p) > 2 and not any(x in p for x in ["АО", "ТОО", "ИП", "ООО"]):
                if p not in persons:
                    persons.append(p)

        # 📅 Даты
        for m in DATE_RX.finditer(text):
            dt = m.group(1)
            if dt not in dates:
                dates.append(dt)

        # 💰 Суммы
        for m in MONEY_RX.finditer(text):
            amt = m.group(0)
            if amt not in amounts:
                amounts.append(amt)

        # ⚡ События по ключевым словам
        for sent in _split_sentences(text):
            if any(h in sent.lower() for h in EVENT_HINTS):
                if sent not in [f["text"] for f in facts]:
                    facts.append(
                        {
                            "fact_id": f"f{fact_id}",
                            "type": "event",
                            "text": sent.strip()[:500],
                            "confidence": _conf_from_signal(sent),
                            "sources": [{"file_id": file_id, "page": page}],
                        }
                    )
                    fact_id += 1

    # Если событий нет, но есть базовый контекст — собираем краткую сводку
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
                    "sources": sources[:1] if sources else [],
                }
            )

    return facts, persons, dates, amounts, _dedup_sources(sources)


# ============================================================
# 🧠 Обогащение фактов ролями / действиями
# ============================================================

def enrich_facts_with_roles(facts: list[dict]) -> list[dict]:
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
        "организовал", "заключил договор", "получил доступ", "снял деньги"
    ]

    for f in facts:
        txt = f["text"].lower()
        f["role"] = next((r for k, r in ROLE_HINTS.items() if k in txt), "неопределено")
        f["action"] = next((a for a in ACTION_HINTS if a in txt), None)
        f["time"] = next(
            (d for d in re.findall(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}", txt)), None
        )
    return facts


# ============================================================
# 🔗 Группировка фактов по типам лиц (для будущего анализа)
# ============================================================

def group_facts_by_entities(facts: list[dict]) -> dict:
    groups = {
        "следователи": defaultdict(list),
        "потерпевшие": defaultdict(list),
        "прочие": [],
    }

    for fact in facts:
        text = fact.get("text", "").lower()
        if any(name in text for name in ["закиев", "шакенов", "дженалинов"]):
            groups["следователи"]["Следователь"].append(fact)
        elif any(name in text for name in ["нуркимбаев", "беков", "кох", "кусаинов"]):
            groups["потерпевшие"]["Потерпевший"].append(fact)
        else:
            groups["прочие"].append(fact)
    return groups


# ============================================================
# 🧱 Проверка наличия данных о подозреваемом (используется в API)
# ============================================================

def validate_facts_completeness(docs: list[Dict[str, Any]]):
    """
    Быстрая проверка: есть ли ВООБЩЕ факты и упоминание подозреваемого.
    В API сюда передаются raw-docs, а не facts — это нормально: мы смотрим по text.
    """
    if not docs:
        raise HTTPException(
            status_code=400,
            detail="❌ Не найдены документы для анализа. Проверьте, загружены ли файлы по делу.",
        )

    has_suspect = any(
        "подозреваем" in (d.get("text") or "").lower() for d in docs
    )
    if not has_suspect:
        raise HTTPException(
            status_code=404,
            detail="❌ В текстах не обнаружены сведения о подозреваемом. "
                   "Требуется проверить OCR и полноту загруженных материалов.",
        )


# ============================================================
# ✅ Проверка полноты по ст. 204 УПК РК
# ============================================================

def _check_204_completeness(
    facts,
    persons,
    dates,
    amounts,
    roles=None,
    events=None,
    legal_facts=None,
    timeline=None,
):
    roles = roles or {}
    events = events or []
    legal_facts = legal_facts or {}

    def present(x):
        return bool(x)

    checklist = [
        {
            "item": "Установлена личность подозреваемого",
            "present": present(roles.get("suspect")),
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
        "enough_for_draft": len(missing) <= 2,
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
# 🧾 Построение раздела «УСТАНОВИЛ» (fallback-версия)
# ============================================================

def _build_ustanovil_text(
    facts: List[dict],
    sources: List[dict],
    completeness: dict,
) -> str:
    if not facts:
        return "Существенных фактов не обнаружено. [ТРЕБУЕТ ПРОВЕРКИ]"

    lines: List[str] = []
    for f in facts:
        src_str = _src_str(f.get("sources"))
        conf = f.get("confidence", 0.5)
        suffix = "" if conf >= 0.75 else " [ТРЕБУЕТ ПРОВЕРКИ]"
        lines.append(f"— {f['text']} {src_str} (уверенность={conf:.2f}){suffix}")

    if completeness.get("missing"):
        lines.append("")
        lines.append("Недостающие элементы для полной квалификации по ст. 204 УПК РК:")
        for m in completeness["missing"]:
            lines.append(f"• {m}")

    return "\n".join(lines)


# ============================================================
# 🧾 Простое построение Постановления (fallback, без LLM)
# ============================================================

def _build_postanovlenie_simple(
    city: str,
    date_str: str,
    investigator_line: str,
    case_id: Optional[str],
    ustanovil_text: str,
    mentioned_articles: List[Dict[str, Any]],
    completeness: dict,
    investigator_fio: str = ""
) -> str:

    rus_date = _rus_date(date_str)

    # список упоминаний статей
    if mentioned_articles:
        arts = sorted({f"{a.get('code', '')} ст.{a.get('article', '?')}" for a in mentioned_articles})
        arts_line = "Упоминания статей: " + "; ".join(arts)
    else:
        arts_line = "Упоминаний статей нет."

    # решение
    if completeness.get("enough_for_draft"):
        decision = "Квалифицировать деяние подозреваемого по соответствующим статьям УК Республики Казахстан."
    else:
        decision = "Окончательную квалификацию определить после получения недостающих материалов."

    return f"""
ПОСТАНОВЛЕНИЕ
о квалификации деяния подозреваемого

{city}, {rus_date}

Материалы дела № {case_id}
{arts_line}

УСТАНОВИЛ:
{ustanovil_text}

ПОСТАНОВИЛ:
{decision}

Подпись:
Следователь: {investigator_line}
ФИО: {investigator_fio}
______________________
Дата: {rus_date}

Права подозреваемого, предусмотренные ст. 64 УПК РК:
- право знать, в чём он подозревается;
- право давать объяснения или отказаться от дачи объяснений;
- право на защитника;
- право представлять доказательства;
- право заявлять ходатайства;
- право обжаловать действия и решения органа расследования.

Черновик сформирован автоматически; окончательное решение принимает следователь после проверки и утверждения прокурором.
""".strip()

def _legal_fact_filter(fact_text: str) -> bool:
    """
    Фильтр юридически значимых фактов.
    True  → факт относится к преступлению (оставляем).
    False → служебный/мусорный/процессуальный факт (выбрасываем).
    """
    if not fact_text:
        return False

    t = fact_text.lower().strip()

    # ❌ 1. QR-коды, ЭЦП, хэши, технич. данные PDF
    blocked_pdf = [
        "qr", "хеш", "хэш", "ecp", "эцп", "электронный документ",
        "подписал", "подписано", "подписан", "подписано следователем",
        "данные эцп", "код", "подготовил", "подготовлено",
        "копия постановления", "получена копия", "получил копию",
        "электронный pdf", "время подписания", "дата подписания",
    ]
    if any(w in t for w in blocked_pdf):
        return False

    # ❌ 2. Разъяснение прав / обязанностей (потерпевший, гражданский истец и т.п.)
    if "разъясн" in t and ("прав" in t or "обязан" in t):
        return False

    # ❌ 3. Обязанности потерпевшего, правила поведения
    if "явиться по вызову" in t or "обязан" in t:
        return False

    # ❌ 4. Технические средства видеозаписи
    if "видеокамера" in t or "iphone" in t or "sony" in t:
        return False

    # ❌ 5. Руководители СОГ / СУ ДЭР (служебная информация)
    if "руководител" in t and ("сог" in t or "дер" in t or "су дер" in t):
        return False

    # ❌ 6. Пауза / ведение видеозаписи
    if "видеосъем" in t or "видеозапись" in t or "приостанов" in t:
        return False

    # ❌ 7. Общие служебные документы (если нет показаний)
    blocked_docs = ["постановление", "протокол", "уведомление"]
    if any(w in t for w in blocked_docs) and "показан" not in t and "допрос" not in t:
        # исключение: протокол допроса / показаний — оставить
        return False

    # ❌ 8. Пустые / слишком короткие факты
    if len(t) < 30:
        return False

    # ✔ 9. Показания потерпевших
    if "потерпев" in t and ("показал" in t or "показан" in t or "показани" in t):
        return True

    # ✔ 10. Признаки пирамиды / TACORP / движения денег / ущерба
    keywords_crime = [
        "tacorp", "таcorp", "пирамид", "вовлечен", "вступил", "привлек",
        "денежн", "перевел", "перевёл", "получил деньги", "ущерб", "средства",
        "финансов", "экономическ", "экспертиз",
    ]
    if any(w in t for w in keywords_crime):
        return True

    # ✔ 11. Любые действия лица, связанные с преступлением
    action_words = ["соверш", "действ", "организ", "руковод", "получ", "присво", "обманул"]
    if any(w in t for w in action_words):
        return True

    # ❌ Всё остальное — выбрасываем
    return False


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

    # 0️⃣ Базовые проверки
    if not docs:
        logger.warning("⚠️ Нет документов для анализа")
        return _empty_result(case_id, "Документы для анализа отсутствуют")

    if not date_str:
        date_str = datetime.now().strftime("%d.%m.%Y")

    try:
        # 1️⃣ Извлечение фактов / сущностей из docs
        try:
            facts, persons, dates, amounts, sources = _extract_facts_and_sources(docs)
        except Exception as e:
            raise RuntimeError(f"_extract_facts_and_sources error: {e}")

        facts = enrich_facts_with_roles(facts)

        # 1.1 Фильтрация мусорных фактов
        raw_count = len(facts)
        facts = [f for f in facts if _legal_fact_filter(f.get("text", ""))]
        logger.info(f"ФИЛЬТР ФАКТОВ: было={raw_count}, после фильтра={len(facts)}")

        # 2️⃣ Глубокий EXTRACTOR (роли, события, хронология, юр. признаки)
        try:
            extracted = extract_all(facts, persons, dates, amounts)
            roles = extracted.get("roles", {})
            events = extracted.get("events", [])
            timeline = extracted.get("timeline", [])
            legal_facts = extracted.get("legal_facts", {})
            logger.info("📌 EXTRACTOR: roles/events/timeline/legal_facts получены")
        except Exception as e:
            logger.error(f"Ошибка EXTRACTOR: {e}")
            roles, events, timeline, legal_facts = {}, [], [], {}

        # 3️⃣ Проверка полноты по ст. 204 УПК РК
        completeness = _check_204_completeness(
            facts=facts,
            persons=persons,
            dates=dates,
            amounts=amounts,
            roles=roles,
            events=events,
            legal_facts=legal_facts,
            timeline=timeline,
        )

        # 4️⃣ Извлечение статей и подготовка контекста АФМ
        mentioned_articles = _extract_articles(docs)
        logger.info(f"Упоминаний статей: {len(mentioned_articles)}")

        law_contexts: List[str] = []
        for art in mentioned_articles:
            num = art.get("article")
            if num and num in ALL_AFM_LAWS:
                law_contexts.append(_resolve_law_context(num))
        law_context_text = "\n".join(law_contexts[:5]) if law_contexts else ""

        # 5️⃣ Базовый раздел «УСТАНОВИЛ» (fallback)
        ustanovil_text = _build_ustanovil_text(facts, sources, completeness)

        # 6️⃣ Попытка улучшить «УСТАНОВИЛ» через LLM
        if facts:
            try:
                fact_lines: List[str] = []
                for f in facts:
                    fact_lines.append(f"- {f['text']} {_src_str(f.get('sources'))}")

                missing_text = ", ".join(completeness.get("missing", [])) or "нет"

                strict_prompt = prompts.U_STSTRICT.format(
                    facts="\n".join(fact_lines),
                    missing=missing_text,
                )

                system_prompt = (
                    "Ты — специализированный юридический модуль «AI_Qualifier» "
                    "для органов финансового мониторинга. "
                    "Строго следуй фактам, не придумывай новых сведений. "
                    "Нельзя включать в раздел «УСТАНОВИЛ» технические данные о QR-кодах, ЭЦП, "
                    "видеокамерах, служебных назначениях и разъяснения прав; "
                    "описывай только картину преступления (кто, что сделал, когда, где, каким способом, "
                    "какой ущерб, связь с TACORP/пирамидой и движением денег)."
                )

                ustanovil_text = _ask_llm(
                    prompt=strict_prompt,
                    system_prompt=system_prompt,
                )
                logger.info("Раздел «УСТАНОВИЛ» улучшен через LLM.")
            except LLMUnavailableError as e:
                logger.warning(f"LLM недоступен для раздела «УСТАНОВИЛ»: {e}")

        # 7️⃣ Генерация финального Постановления
        final_postanovlenie: str

        try:
            safe_article = (
                mentioned_articles[0].get("article", "[Требует уточнения]")
                if mentioned_articles else "[Требует уточнения]"
            )

            post_prompt = prompts.P_POST.format(ustanovil=ustanovil_text)

            system_for_post = (
                "Ты — опытный старший следователь АФМ с юридическим образованием. "
                "Составь проект процессуального документа "
                "«Постановление о квалификации деяния подозреваемого» "
                "по требованиям УПК РК. "
                "Не добавляй фактов, которых нет в разделе «УСТАНОВИЛ». "
                "В тексте не дублируй документ дважды, используй единственный заголовок "
                "и структуру: шапка → УСТАНОВИЛ → ПОСТАНОВИЛ → при необходимости подпись."
            )

            full_user_prompt = f"""
Материалы дела № {case_id}.
Место вынесения: {city}.
Дата: {_rus_date(date_str)}.

Вспомогательный юридический контекст (если есть):
{law_context_text or "[нет дополнительных формулировок законов]"}

Основывайся исключительно на разделе «УСТАНОВИЛ» ниже.

{post_prompt}

Статья для основной квалификации (если это подтверждается фактами): {safe_article}.
Если данных недостаточно — в разделе «ПОСТАНОВИЛ» отрази необходимость получения
дополнительных доказательств перед окончательной квалификацией.
"""

            final_postanovlenie = _ask_llm(
                prompt=full_user_prompt,
                system_prompt=system_for_post,
            )
            logger.info("Постановление сгенерировано через LLM.")
        except LLMUnavailableError as e:
            logger.warning(f"LLM недоступен для Постановления: {e}")
            final_postanovlenie = _build_postanovlenie_simple(
                city=city,
                date_str=date_str,
                investigator_line=investigator_line,
                case_id=case_id,
                ustanovil_text=ustanovil_text,
                mentioned_articles=mentioned_articles,
                completeness=completeness,
                investigator_fio=investigator_fio,
            )

        # 8️⃣ Страховка структуры (наличие «УСТАНОВИЛ» и «ПОСТАНОВИЛ»)
        lower_body = final_postanovlenie.lower()
        if "установил" not in lower_body or "постановил" not in lower_body:
            logger.warning("⚠️ LLM отклонился от структуры, применяю fallback-шаблон Постановления.")
            final_postanovlenie = _build_postanovlenie_simple(
                city=city,
                date_str=date_str,
                investigator_line=investigator_line,
                case_id=case_id,
                ustanovil_text=ustanovil_text,
                mentioned_articles=mentioned_articles,
                completeness=completeness,
                investigator_fio=investigator_fio,
            )

        # 9️⃣ Уверенность + базовый result
        overall_conf = _overall_confidence(facts, completeness)
        warnings: List[str] = []

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
            "established_text": ustanovil_text.strip(),
            "final_postanovlenie": final_postanovlenie.strip(),
            "sources": sources,
            "confidence": round(overall_conf, 3),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "investigator_fio": investigator_fio,
            "investigator_line": investigator_line,
            "warnings": warnings,
        }

        # 🔟 Anti-hallucination верификация
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

        logger.info(
            f"✅ Квалификация завершена. verdict={result.get('verdict')}, "
            f"conf={result.get('confidence'):.2f}"
        )
        return result

    except LLMUnavailableError as e:
        logger.error(f"LLMUnavailableError в qualify_documents: {e}")
        raise
    except Exception as e:
        logger.error(f"Критическая ошибка квалификации: {e}", exc_info=True)
        return _empty_result(case_id, f"Ошибка обработки: {str(e)}")

# ============================================================
# 🔹 Fallback-результат при ошибках
# ============================================================

def _empty_result(case_id: str, msg: str, investigator_fio: str = "", investigator_line: str = "") -> dict:
    return {
        "generation_id": None,
        "model_version": MODEL_VERSION,
        "case_id": case_id,
        "established_text": "",
        "final_postanovlenie": f"[ОШИБКА]: {msg}",
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
    }


# ============================================================
# 🧰 Вспомогательные утилиты
# ============================================================

def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.!\?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 10]


def _context_snippet(text: str, start: int, end: int, radius: int = CONTEXT_RADIUS) -> str:
    a, b = max(0, start - radius), min(len(text), end + radius)
    return text[a:b].replace("\n", " ").strip()


def _src_str(sources: List[Dict[str, Any]] | None) -> str:
    if not sources:
        return "[источник: не указан]"
    show = [f"[{s.get('file_id', '?')}:{s.get('page', '-')}]"
            for s in sources[:3]]
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


def _overall_confidence(facts: List[dict], completeness: dict) -> float:
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
