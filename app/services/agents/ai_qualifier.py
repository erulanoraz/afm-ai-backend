# app/services/agents/ai_qualifier.py
from __future__ import annotations

import logging
import uuid
import json
import re
from collections import Counter
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.services.llm_client import LLMClient
from app.services.facts.fact_models import LegalFact
from app.services.facts.fact_tokenizer import FactTokenizer
from app.services.facts.fact_graph import FactGraph
from app.services.facts.fact_filter import FactFilter
from app.services.rag_router import RAGRouter
from app.services.validation.verifier import (
    run_full_verification,
    verify_sentence_token_alignment,
)
from app.services.agents import prompts
from app.services.agents.crime_classifier import (
    classify_by_tokens,
    format_classification_debug,
)
from app.utils.sentence_splitter import split_into_sentences
from app.utils.utils_v4 import validate_docs

logger = logging.getLogger(__name__)

llm = LLMClient()

# ВЕРСИЮ ОБНОВИЛИ, ЧТОБЫ ВИДНО БЫЛО, ЧТО ЛОГИКА ПЕРЕРАБОТАНА
MODEL_VERSION = "qualifier-llm-6.0.2"


# ============================================================
# 🧠 Вспомогательные функции LLM
# ============================================================

def ask_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Обёртка над LLMClient с логированием и защитой от падений.
    """
    try:
        resp = llm.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        if resp is None:
            logger.error("LLM ERROR: ответ None")
            return "[LLM_ERROR]"
        if isinstance(resp, dict):
            # Если LLMClient вернул dict (стиль OpenAI), пробуем вытащить текст
            try:
                content = resp["choices"][0]["message"]["content"]
                return (content or "").strip()
            except Exception:
                logger.error(f"LLM ERROR: неожиданный формат dict-ответа: {resp}")
                return "[LLM_ERROR]"
        return str(resp).strip()
    except Exception as e:
        logger.error(f"LLM ERROR: {e}")
        return "[LLM_ERROR]"


# ============================================================
# 🔧 Устойчивый JSON-парсер (с авто-восстановлением)
# ============================================================

def safe_json_loads(raw: str) -> Optional[dict]:
    """
    JSON Recovery Layer — AI_Qualifier 6.0+
    Исправляет частично сломанные JSON-структуры от LLM.

    Поддерживает:
    - удаление ```json ... ``` оболочек;
    - удаление лишних запятых перед ] и };
    - добивание недостающих скобок;
    - поиск первой корректной { ... } структуры внутри текста.
    """
    if not raw:
        return None

    cleaned = raw.strip()

    # 1) удаляем markdown-оболочку ```json ... ```
    cleaned = re.sub(r"```[a-zA-Z0-9]*", "", cleaned).strip("` \n\r\t")

    # 2) удаляем возможное слово "json" в начале
    cleaned = re.sub(r"^json\s*", "", cleaned, flags=re.IGNORECASE).strip()

    # 3) убираем висящие запятые перед закрывающими скобками
    cleaned = re.sub(r",\s*]", "]", cleaned)
    cleaned = re.sub(r",\s*}", "}", cleaned)

    # 4) добиваем недостающие фигурные скобки
    open_braces = cleaned.count("{")
    close_braces = cleaned.count("}")
    if open_braces > close_braces:
        cleaned += "}" * (open_braces - close_braces)

    # 5) добиваем недостающие квадратные скобки
    open_arr = cleaned.count("[")
    close_arr = cleaned.count("]")
    if open_arr > close_arr:
        cleaned += "]" * (open_arr - close_arr)

    # 6) первая попытка parse
    try:
        parsed = json.loads(cleaned)
        # ожидаем dict; если LLM вернул массив с одним объектом — берём первый
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 7) fallback: вытащить первую {...} структуру
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    return None


# ============================================================
# 🧠 Извлечение token_id
# ============================================================

def _extract_token_ids_from_fact(fact: LegalFact) -> List[str]:
    """
    Унифицированное извлечение token_id/token_ids из LegalFact,
    чтобы не ловить 'method' object is not iterable.
    """
    token_ids: List[str] = []

    # Вариант 1: единственный token_id
    single = getattr(fact, "token_id", None)
    if isinstance(single, str) and single:
        token_ids.append(single)

    # Вариант 2: поле или метод token_ids
    attr = getattr(fact, "token_ids", None)
    if attr:
        try:
            value = attr() if callable(attr) else attr
            if isinstance(value, (list, tuple, set)):
                for v in value:
                    if isinstance(v, str) and v:
                        token_ids.append(v)
        except Exception as e:
            logger.warning(
                f"⚠ Не удалось извлечь token_ids из факта {getattr('id', None)}: {e}"
            )

    # убираем дубликаты
    return list(sorted(set(token_ids)))


# ============================================================
# 🔧 Auto-clean routed facts (person-only, мусорные имена)
# ============================================================

_BAD_PERSON_TOKENS = {
    "после",
    "кроме",
    "далее",
    "не",
    "нет",
    "вы",
    "о",
    "а",
    "для",
    "примерно",
    "назад",
    "однако",
}


def _cleanup_routed_facts(facts: List[LegalFact]) -> List[LegalFact]:
    """
    Удаляет:
    - факты, состоящие только из person-токенов;
    - person-токены с "служебными" значениями (После, Кроме, Вы, Не, Нет и т.п.).
    """
    cleaned: List[LegalFact] = []

    for f in facts:
        tokens = getattr(f, "tokens", []) or []
        persons = [t for t in tokens if t.type == "person"]
        other_tokens = [t for t in tokens if t.type != "person"]

        # Полностью пропускаем факты, где только "person" и нет других сущностей
        if persons and not other_tokens:
            continue

        # Чистим "плохие" person-значения
        filtered_tokens = []
        for t in tokens:
            if t.type == "person":
                if t.value and t.value.strip().lower() in _BAD_PERSON_TOKENS:
                    continue
            filtered_tokens.append(t)

        f.tokens = filtered_tokens
        cleaned.append(f)

    return cleaned


# ============================================================
# 🔧 Валидация фактов перед LLM
# ============================================================

def _validate_facts_for_llm(facts: List[LegalFact]) -> List[LegalFact]:
    """
    Отбрасывает пустые / странные факты перед отправкой в LLM.
    """
    valid: List[LegalFact] = []
    for f in facts:
        tokens = getattr(f, "tokens", None)
        if not tokens or not isinstance(tokens, list):
            continue
        if len(tokens) == 0:
            continue
        valid.append(f)
    return valid


# ============================================================
# 🔧 Нормализация ФИО и имени проекта/организации
# ============================================================

def _normalize_person_name(name: str) -> str:
    """
    Унифицирует ФИО: убирает лишние пробелы, приводит к аккуратному виду.
    Не лезет в логику пола/падежей — только формат.
    """
    if not name:
        return ""
    n = re.sub(r"\s+", " ", name).strip()
    if not n:
        return ""
    parts = n.split(" ")
    return " ".join(p[:1].upper() + p[1:] for p in parts)


def _normalize_project_name(name: str) -> str:
    """
    Унифицирует название проекта/организации/платформы:
    убирает кавычки, лишние пробелы, сохраняет общий вид.
    """
    if not name:
        return ""
    n = name.strip().strip("«»\"'“”„“")
    n = re.sub(r"\s+", " ", n)
    return n.strip()


# ============================================================
# 🔧 Сбор мета-информации по делу
#      (suspects, victims, organizations, platforms, amounts)
# ============================================================

def _collect_case_meta(facts: List[LegalFact]) -> Dict[str, Any]:
    """
    Собирает метаданные дела на основе LegalFact:
    - project_name
    - suspects (ФИО)
    - victims (ФИО)
    - organizations (названия)
    - platforms (названия)
    - all_persons
    - amounts_summary (min/max/total по числам из amount)
    - participants_formatted (формат 2: «лицо, указанное в материалах как ...»)
    """
    suspects: set[str] = set()
    victims: set[str] = set()
    all_persons: set[str] = set()
    organizations: set[str] = set()
    platforms: set[str] = set()
    project_candidates: List[str] = []
    amount_values: List[int] = []

    for f in facts:
        txt_raw = getattr(f, "text", "") or ""
        txt = txt_raw.lower()
        tokens = getattr(f, "tokens", []) or []
        role = (getattr(f, "role", "") or "").lower()

        # role_label токены (victim/suspect/organizer/witness ...)
        role_labels = {t.value for t in tokens if t.type == "role_label" and t.value}

        # PERSONS
        persons_in_fact = [t.value for t in tokens if t.type == "person" and t.value]
        norm_persons: List[str] = []
        for p in persons_in_fact:
            n = _normalize_person_name(p)
            if n:
                norm_persons.append(n)
                all_persons.add(n)

        # Heuristics для подозреваемых
        is_suspect_fact = False
        if role in ("suspect_action", "fraud_action", "fraud_event"):
            is_suspect_fact = True
        if "подозреваем" in txt:
            is_suspect_fact = True
        if any("suspect" in str(lbl).lower() for lbl in role_labels):
            is_suspect_fact = True

        if is_suspect_fact:
            for p in norm_persons:
                suspects.add(p)

        # Heuristics для потерпевших
        is_victim_fact = False
        if "потерпевш" in txt:
            is_victim_fact = True
        if any("victim" in str(lbl).lower() for lbl in role_labels):
            is_victim_fact = True

        if is_victim_fact:
            for p in norm_persons:
                victims.add(p)

        # AMOUNTS
        for t in tokens:
            if t.type == "amount" and t.value:
                digits = re.sub(r"[^\d]", "", t.value)
                if digits:
                    try:
                        amount_values.append(int(digits))
                    except Exception:
                        pass

        # ORGANIZATIONS / PROJECTS / PLATFORMS — через токены
        for t in tokens:
            t_type = getattr(t, "type", None)
            t_val = getattr(t, "value", None) or ""
            if not t_type or not t_val:
                continue

            if t_type in ("project", "project_name"):
                name_norm = _normalize_project_name(t_val)
                if name_norm:
                    project_candidates.append(name_norm)

            if t_type in ("organization", "company"):
                name_norm = _normalize_project_name(t_val)
                if name_norm:
                    organizations.add(name_norm)
                    project_candidates.append(name_norm)

            if t_type == "platform":
                name_norm = _normalize_project_name(t_val)
                if name_norm:
                    platforms.add(name_norm)

        # ORGANIZATIONS — через текстовые шаблоны
        for m in re.findall(
            r"(проект|компания|организация)\s+«([^»]{2,80})»",
            txt_raw,
            flags=re.IGNORECASE,
        ):
            name_norm = _normalize_project_name(m[1])
            if name_norm:
                organizations.add(name_norm)
                project_candidates.append(name_norm)

        # PLATFORMS — через текстовые шаблоны
        for m in re.findall(
            r"(платформа|система)\s+«([^»]{2,80})»",
            txt_raw,
            flags=re.IGNORECASE,
        ):
            name_norm = _normalize_project_name(m[1])
            if name_norm:
                platforms.add(name_norm)

    project_name = None
    if project_candidates:
        freq = Counter(project_candidates)
        project_name = freq.most_common(1)[0][0]

    amounts_summary = None
    if amount_values:
        try:
            amounts_summary = {
                "count": len(amount_values),
                "min": min(amount_values),
                "max": max(amount_values),
                "total": sum(amount_values),
            }
        except Exception:
            amounts_summary = {
                "count": len(amount_values),
            }

    # Формат 2: юридически безопасные описания участников
    participants_formatted: Dict[str, List[str]] = {}

    if suspects:
        participants_formatted["suspects"] = [
            f"подозреваемый, указанный в материалах как {s}"
            for s in sorted(suspects)
        ]

    if victims:
        participants_formatted["victims"] = [
            f"потерпевший, указанный в материалах как {v}"
            for v in sorted(victims)
        ]

    if organizations:
        participants_formatted["organizations"] = [
            f"организация, фигурирующая в материалах как «{o}»"
            for o in sorted(organizations)
        ]

    if platforms:
        participants_formatted["platforms"] = [
            f"платформа, обозначенная в материалах как «{p}»"
            for p in sorted(platforms)
        ]

    meta: Dict[str, Any] = {
        "project_name": project_name,
        "suspects": sorted(suspects),
        "victims": sorted(victims),
        "organizations": sorted(organizations),
        "platforms": sorted(platforms),
        "victims_count": len(victims),
        "all_persons": sorted(all_persons),
        "amounts_summary": amounts_summary,
        "participants_formatted": participants_formatted,
    }

    return meta


# ============================================================
# 🔧 Строгий sentence/token alignment поверх базового
# ============================================================

def _strict_sentence_token_alignment(
    sentence_map: List[Dict[str, Any]],
    used_tokens: List[str],
    all_token_ids: List[str],
) -> Dict[str, Any]:
    """
    Дополнительный строгий контроль:
    - все tokens, на которые ссылается LLM, должны быть в all_token_ids;
    - alignment_ok = False, если есть неизвестные токены.
    """
    all_set = set(all_token_ids)
    used_set = set(used_tokens)

    unknown = sorted(list(used_set - all_set))
    missing = sorted(list(all_set - used_set))

    return {
        "unknown_tokens": unknown,
        "missing_tokens": missing,
        "alignment_ok": len(unknown) == 0,
    }


# ============================================================
# 🔧 Очистка технических вставок (token-id, UUID и т.п.)
# ============================================================

_TECH_TOKEN_RE = re.compile(r"\(token [^)]+\)", re.IGNORECASE)
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_TOKEN_WORD_UUID_RE = re.compile(r"token\s+[0-9a-fA-F\-]{8,}", re.IGNORECASE)


def _strip_technical_tokens(text: str) -> str:
    """
    Убирает из текста служебные конструкции:
    • '(token XXXXX-...)'
    • чистые UUID
    • фразы вида 'token XXXXX-...'
    Используется как защитный слой, чтобы в финальном документе
    не было внутренних идентификаторов.
    """
    if not text:
        return text

    cleaned = _TECH_TOKEN_RE.sub("", text)
    cleaned = _UUID_RE.sub("", cleaned)
    cleaned = _TOKEN_WORD_UUID_RE.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


# ============================================================
# 🔍 Авто-определение города по текстам
# ============================================================

def _detect_city_from_docs(docs: List[dict]) -> str:
    """
    Очень мягкое определение города:
    - ищем г. Алматы, Астана, Шымкент, Павлодар, Караганда, Костанай, Актау, Актобе и др.
    - если нет — возвращаем пусто
    """
    if not docs:
        return ""

    cities = [
        "Алматы",
        "Астана",
        "Нур-Султан",
        "Шымкент",
        "Павлодар",
        "Караганда",
        "Костанай",
        "Актау",
        "Актобе",
        "Тараз",
        "Усть-Каменогорск",
        "Семей",
        "Кокшетау",
    ]

    merged_text = " ".join((d.get("text") or "").lower() for d in docs)

    for c in cities:
        if c.lower() in merged_text:
            return c

    return ""


def _count_words(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


# ============================================================
# ⭐ Основная функция квалификации
# ============================================================

def qualify_documents(
    case_id: Optional[str],
    docs: List[Dict[str, Any]],
    city: Optional[str] = None,
    investigator_fio: str = "Не указан",
    investigator_line: str = "Следователь",
    date_str: Optional[str] = None,
) -> Dict[str, Any]:

    # базовая валидация входных документов
    validate_docs(docs)

    if not date_str:
        date_str = datetime.now().strftime("%d.%m.%Y")

    logger.info(
        f"▶️ QUALIFIER 6.0.2 (token-json): docs={len(docs)}, case_id={case_id or '-'}"
    )

    # ------------------------------------------------------------
    # 0) Автоматическое определение города
    # ------------------------------------------------------------
    auto_city = _detect_city_from_docs(docs)
    if auto_city:
        city = auto_city
    else:
        city = city or ""

    # =====================================================================
    # 1) Tokenizer
    # =====================================================================
    tokenizer = FactTokenizer()
    tokenized_facts: List[LegalFact] = tokenizer.tokenize(docs)
    logger.info(f"📘 Tokenizer: извлечено фактов = {len(tokenized_facts)}")

    if not tokenized_facts:
        return _empty_result(
            case_id,
            "Факты не обнаружены после токенизации.",
            investigator_fio,
            investigator_line,
        )

    # =====================================================================
    # 2) FactGraph (merge)
    # =====================================================================
    graph = FactGraph()
    merged: List[LegalFact] = graph.build(tokenized_facts)
    logger.info(f"📗 FactGraph: после объединения = {len(merged)}")

    if not merged:
        return _empty_result(
            case_id,
            "После объединения фактов (FactGraph) ничего не осталось.",
            investigator_fio,
            investigator_line,
        )

    # =====================================================================
    # 2.1) FactFilter — очистка процессуалки и мусора
    # =====================================================================
    fact_filter = FactFilter()
    filtered_facts: List[LegalFact] = fact_filter.filter_for_qualifier(merged)
    logger.info(f"📙 FactFilter: после фильтрации = {len(filtered_facts)}")

    if not filtered_facts:
        return _empty_result(
            case_id,
            "Нет релевантных фактов для квалификации после фильтрации.",
            investigator_fio,
            investigator_line,
        )

    # =====================================================================
    # 2.2) Pre-crime classification (чисто диагностическая)
    # =====================================================================
    pre_cls_input = [f for f in filtered_facts if getattr(f, "role", "") != "generic_fact"]
    if not pre_cls_input:
        pre_cls_input = filtered_facts

    pre_classification = classify_by_tokens(pre_cls_input)
    logger.info(
        "⚖ Pre-crime classification (для отладки):\n"
        + format_classification_debug(pre_classification)
    )

    # =====================================================================
    # 3) RAG Router (БЕЗ target_article — универсальный режим)
    # =====================================================================
    router = RAGRouter()
    routed_facts: List[LegalFact] = router.route_for_qualifier(
        filtered_facts,
        target_article=None,  # НЕ навязываем состав, роутер работает универсально
    )
    logger.info(f"📙 RAG Router: кандидатов (сырой вывод) = {len(routed_facts)}")

    if not routed_facts:
        return _empty_result(
            case_id,
            "RAG Router не нашёл фактов для квалификации.",
            investigator_fio,
            investigator_line,
        )

    # 3.1) Auto-clean routed facts (убираем person-only мусор)
    routed_facts = _cleanup_routed_facts(routed_facts)
    routed_facts = _validate_facts_for_llm(routed_facts)

    if not routed_facts:
        return _empty_result(
            case_id,
            "После авто-чистки фактов не осталось для квалификации.",
            investigator_fio,
            investigator_line,
        )

    # 3.2) Группировка по routing_group (primary / secondary / reserve)
    primary_facts: List[LegalFact] = []
    secondary_facts: List[LegalFact] = []
    reserve_facts: List[LegalFact] = []

    for f in routed_facts:
        grp = getattr(f, "routing_group", None)
        if grp == "secondary":
            secondary_facts.append(f)
        elif grp == "reserve":
            reserve_facts.append(f)
        else:
            primary_facts.append(f)

    routed_facts = primary_facts + secondary_facts + reserve_facts

    logger.info(
        "📙 RAG Router: группировка после авто-чистки → "
        f"primary={len(primary_facts)}, "
        f"secondary={len(secondary_facts)}, "
        f"reserve={len(reserve_facts)}, "
        f"total={len(routed_facts)}"
    )

    # 3.3) Сбор мета-информации по делу (project_name, suspects, victims, суммы, организации, платформы)
    case_meta = _collect_case_meta(routed_facts)
    logger.info(f"📌 Case meta: {case_meta}")

    # ============================================================
    # 3.4) Crime Classification (финальная, по routed_facts)
    #      — чисто для auto_classification, НЕ навязываем LLM статьи
    # ============================================================
    cls_input = [f for f in routed_facts if getattr(f, "role", "") != "generic_fact"]
    if not cls_input:
        cls_input = routed_facts

    classification = classify_by_tokens(cls_input)

    logger.info(
        "⚖ Crime classification (финальная по routed_facts):\n"
        + format_classification_debug(classification)
    )

    primary_article = classification.get("primary")
    secondary_articles = classification.get("secondary", []) or []

    # Универсальный список всех выявленных статей (чтобы 217 НЕ выглядела «отдельно»)
    articles_all: List[str] = []
    if primary_article:
        articles_all.append(primary_article)
    for a in secondary_articles:
        if a and a not in articles_all:
            articles_all.append(a)

    logger.info(
        f"⚖ Итоговая авто-классификация: primary={primary_article}, "
        f"secondary={secondary_articles}, all={articles_all}"
    )

    # =====================================================================
    # 4) Подготовка payload фактов для LLM (JSON strict, с группами)
    # =====================================================================
    facts_payload: List[Dict[str, Any]] = []
    for f in routed_facts:
        d = f.model_dump()

        # для верификатора: гарантируем поле sources, даже если модель называет его иначе
        if "sources" not in d and "source_refs" in d:
            d["sources"] = d.get("source_refs") or []

        # помечаем routing_group, если он есть у факта
        grp = getattr(f, "routing_group", None)
        if grp:
            d["routing_group"] = grp

        facts_payload.append(d)

    logger.info(
        f"📊 Facts payload для LLM: всего={len(facts_payload)}, "
        f"primary≈{sum(1 for x in facts_payload if x.get('routing_group') == 'primary') or len(facts_payload)}, "
        f"secondary={sum(1 for x in facts_payload if x.get('routing_group') == 'secondary')}, "
        f"reserve={sum(1 for x in facts_payload if x.get('routing_group') == 'reserve')}"
    )

    # =====================================================================
    # 5) Вызов LLM для «УСТАНОВИЛ» (P_UST_TOKENS_JSON)
    #    — БЕЗ передачи статей, только факты + meta с участниками
    # =====================================================================
    system_prompt = prompts.P_UST_TOKENS_JSON

    user_payload = {
        "facts": facts_payload,
        "meta": case_meta,  # project_name, suspects, victims, organizations, platforms, суммы, participants_formatted
        # ВАЖНО: НИКАКИХ primary_article / secondary_articles здесь нет.
    }

    user_prompt = json.dumps(user_payload, ensure_ascii=False, indent=2)

    response = ask_llm(system_prompt, user_prompt)

    # =====================================================================
    # 6) Парсинг JSON-ответа LLM по «УСТАНОВИЛ»
    # =====================================================================
    try:
        if response.startswith("[LLM_ERROR]"):
            raise ValueError("LLM returned error marker")

        parsed = safe_json_loads(response)
        if not parsed or not isinstance(parsed, dict):
            raise ValueError("JSON parse failed")

        ustanovil_text = (parsed.get("ustanovil") or "").strip()
        sentence_map = parsed.get("sentences", []) or []
        used_tokens = sorted({t for s in sentence_map for t in s.get("tokens", [])})

        logger.info(
            f"📘 SENTENCE–TOKEN alignment получен: {len(sentence_map)} предложений"
        )
        logger.info(f"📘 USED TOKENS от LLM: {used_tokens}")

    except Exception as e:
        logger.error(f"❌ Некорректный JSON от LLM (USTANOVIL): {e}")
        logger.error(
            f"❌ Сырой ответ LLM (усечён до 1000 символов): {str(response)[:1000]}"
        )
        ustanovil_text = _fallback_ustanovil(routed_facts)
        sentence_map = []

        used_tokens = []
        for f in routed_facts:
            used_tokens.extend(_extract_token_ids_from_fact(f))

    # Если LLM вернул пустой «УСТАНОВИЛ» — fallback по фактам
    if not ustanovil_text:
        ustanovil_text = _fallback_ustanovil(routed_facts)
        if not used_tokens:
            used_tokens = []
            for f in routed_facts:
                used_tokens.extend(_extract_token_ids_from_fact(f))

    # Очистка от технических вставок (token-id, UUID и т.п.)
    ustanovil_text = _strip_technical_tokens(ustanovil_text)

    # ------------------------------------------------------------
    # 6.1. Разбиение «УСТАНОВИЛ» на предложения и статистика
    # ------------------------------------------------------------
    _sentences_plain = split_into_sentences(ustanovil_text)
    logger.info(f"📘 USTANOVIL: разбиение на предложения = {len(_sentences_plain)}")

    ustanovil_word_count = _count_words(ustanovil_text)
    logger.info(f"📘 USTANOVIL: длина ~ {ustanovil_word_count} слов")

    # ------------------------------------------------------------
    # 6.2. Собираем все возможные token_id из фактов
    # ------------------------------------------------------------
    all_token_ids = set()
    for f in routed_facts:
        all_token_ids.update(_extract_token_ids_from_fact(f))

    # ------------------------------------------------------------
    # 6.3. Anti-hallucination: sentence ↔ token alignment
    # ------------------------------------------------------------
    base_alignment = verify_sentence_token_alignment(
        sentence_map=sentence_map,
        used_tokens=list(used_tokens),
        all_token_ids=list(all_token_ids),
    )

    strict_al = _strict_sentence_token_alignment(
        sentence_map=sentence_map,
        used_tokens=list(used_tokens),
        all_token_ids=list(all_token_ids),
    )

    if isinstance(base_alignment, dict):
        alignment = {**base_alignment, **strict_al}
    else:
        alignment = strict_al

    # =====================================================================
    # 7) ПОСТАНОВИЛ — LLM (обычный режим, но JSON-вход)
    #    — статьи УК/УПК ИИ выводит сам из ustanovil_text, мы НЕ подсказываем номера
    # =====================================================================
    post_system = prompts.P_POST
    post_payload = {
        "ustanovil_text": ustanovil_text,
        "meta": case_meta,
        # НЕТ primary_article/secondary_articles — ИИ сам решает, какие статьи указать.
    }
    post_user = json.dumps(post_payload, ensure_ascii=False, indent=2)

    post_text = ask_llm(post_system, post_user)
    if post_text.startswith("[LLM_ERROR]"):
        post_text = _fallback_postanovil(ustanovil_text)

    post_text = _strip_technical_tokens(post_text)

    # =====================================================================
    # 8) Verification (token anti-hallucination + тексты + источники)
    # =====================================================================
    verification = run_full_verification(
        {
            "facts": facts_payload,
            "ustanovil": ustanovil_text,
            "established_text": ustanovil_text,
            "final_postanovlenie": post_text,
            "used_tokens": used_tokens,
            "sentences": sentence_map,
        }
    )

    # =====================================================================
    # 9) Формирование результата
    # =====================================================================
    result = {
        # авто-классификация состава (отдельный слой)
        "auto_classification": classification,
        "primary_article": primary_article,
        "secondary_articles": secondary_articles,
        "articles_all": articles_all,

        # метаданные генерации
        "generation_id": str(uuid.uuid4()),
        "model_version": MODEL_VERSION,
        "case_id": case_id,

        # фактологическая база
        "facts_used": facts_payload,
        "used_tokens": used_tokens,
        "case_meta": case_meta,

        # текст постановления
        "established_text": ustanovil_text.strip(),
        "final_postanovlenie": post_text.strip(),

        # статистика по длине квалификации
        "ustanovil_word_count": ustanovil_word_count,
        "ustanovil_sentence_count": len(_sentences_plain),

        # проверка
        "verification": verification,
        "sentence_map": sentence_map,
        "sentence_alignment": alignment,
        "verification_sentences": verification.get("sentences"),

        # служебные поля
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "investigator_fio": investigator_fio,
        "investigator_line": investigator_line,
        "city": city,
        "date": date_str,
    }

    logger.info(
        f"✔ QUALIFIER 6.0.2 завершён: facts={len(facts_payload)}, used={len(used_tokens)}, "
        f"ustanovil_words={ustanovil_word_count}"
    )
    return result


# ============================================================
# 🔧 Fallback «УСТАНОВИЛ» (формат 2, без мотивов)
# ============================================================

def _fallback_ustanovil(facts: List[LegalFact]) -> str:
    """
    Умный fallback: строит краткую юридическую фабулу из фактов,
    без токенов, без source_refs, без мотивов и оценок.
    Использует формат 2 для участников:
    «лицо, указанное в материалах как ...» / «организация, фигурирующая как ...».
    """

    meta = _collect_case_meta(facts)

    suspects = meta.get("suspects") or []
    victims = meta.get("victims") or []
    organizations = meta.get("organizations") or []
    platforms_named = meta.get("platforms") or []
    project_name = meta.get("project_name")

    persons_other = set()
    amounts = []
    actions = set()
    dates = set()
    platform_flags = set()

    for f in facts:
        txt = (getattr(f, "text", "") or "")
        low = txt.lower()
        tokens = getattr(f, "tokens", []) or []

        for t in tokens:
            if t.type == "person" and t.value:
                v = t.value.strip()
                if len(v) > 2 and v.lower() not in _BAD_PERSON_TOKENS:
                    norm = _normalize_person_name(v)
                    if norm and norm not in suspects and norm not in victims:
                        persons_other.add(norm)

            if t.type == "amount" and t.value:
                amounts.append(t.value)

            if t.type == "date" and t.value:
                dates.add(t.value)

        if any(
            w in low
            for w in [
                "перевел",
                "перевела",
                "отправил",
                "отправила",
                "внес",
                "внесла",
                "пополнил",
                "пополнила",
            ]
        ):
            actions.add("переводами и иными операциями с денежными средствами")

        if "usdt" in low or "okx" in low or "binance" in low:
            platform_flags.add("операциями, связанными с цифровыми сервисами и криптовалютными платформами")

    lines: List[str] = []
    lines.append("По материалам досудебного расследования установлено следующее.")

    # Организация / проект
    org_source_names: List[str] = []
    if project_name:
        org_source_names.append(project_name)
    if organizations:
        for o in organizations:
            if o not in org_source_names:
                org_source_names.append(o)

    if org_source_names:
        main_org = org_source_names[0]
        lines.append(
            f"В материалах фигурирует организация (проект), обозначенная в документах как «{main_org}»."
        )

    # Подозреваемые
    if suspects:
        formatted = ", ".join(
            f"лицо, указанное в материалах как {s}" for s in sorted(suspects)
        )
        lines.append(
            f"В качестве подозреваемых в материалах указаны {formatted}."
        )

    # Потерпевшие
    if victims:
        formatted = ", ".join(
            f"лицо, указанное в материалах как {v}" for v in sorted(victims)
        )
        lines.append(
            f"В материалах отражены потерпевшие, указанные в материалах как {formatted}."
        )

    # Иные участники
    if persons_other:
        lines.append(
            "Дополнительно в материалах упоминаются иные участники, обозначенные в документах как: "
            + ", ".join(sorted(persons_other))
            + "."
        )

    # Действия / операции
    if actions:
        lines.append(
            f"Зафиксированы действия, связанные с {', '.join(sorted(actions))}."
        )

    # Суммы
    if amounts:
        try:
            normalized = []
            for a in amounts:
                digits = re.sub(r"[^\d]", "", a)
                if digits:
                    normalized.append(int(digits))
            if normalized:
                min_v = min(normalized)
                max_v = max(normalized)
                lines.append(
                    f"Операции с денежными средствами отражены на суммы от {min_v} до {max_v} тенге."
                )
        except Exception:
            lines.append(
                "В материалах указаны операции с денежными средствами на следующие суммы: "
                + ", ".join(amounts)
                + "."
            )

    # Платформы по токенам (именованные)
    if platforms_named:
        lines.append(
            "В материалах упомянуты платформы и цифровые сервисы, обозначенные в документах как: "
            + ", ".join(f"«{p}»" for p in sorted(platforms_named))
            + "."
        )

    # Платформы по текстовым признакам
    if platform_flags:
        lines.append(
            "Отмечены также сведения об операциях, связанных с цифровыми сервисами и криптовалютными платформами."
        )

    # Даты
    if dates:
        lines.append(
            "События, изложенные в материалах, соотносятся со следующими датами: "
            + ", ".join(sorted(dates))
            + "."
        )

    # Финальное обобщение без мотивов/выводов
    lines.append(
        "Перечисленные сведения в совокупности характеризуют фактические обстоятельства, "
        "отражённые в материалах досудебного расследования, без оценки их юридической квалификации."
    )

    return " ".join(lines).strip()


def _fallback_postanovil(ustanovil_text: str) -> str:
    return (
        "ПОСТАНОВИЛ:\n"
        "На основании изложенного в разделе «УСТАНОВИЛ» требуется получение дополнительных данных "
        "для окончательной правовой оценки деяния.\n"
    )


# ============================================================
# 🔧 Пустой результат
# ============================================================

def _empty_result(case_id: Optional[str], msg: str, fio: str, line: str) -> Dict[str, Any]:
    return {
        "generation_id": None,
        "model_version": MODEL_VERSION,
        "case_id": case_id,
        "established_text": msg,
        "final_postanovlenie": msg,
        "facts_used": [],
        "used_tokens": [],
        "case_meta": {},
        "verification": {"error": msg, "overall_ok": False},
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "investigator_fio": fio,
        "investigator_line": line,
        "city": None,
        "date": None,
        "ustanovil_word_count": 0,
        "ustanovil_sentence_count": 0,
    }
