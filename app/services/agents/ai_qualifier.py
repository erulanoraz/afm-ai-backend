# app/services/agents/ai_qualifier.py
from __future__ import annotations

import logging
import uuid
import json
import re
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

MODEL_VERSION = "qualifier-llm-4.5.0"


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
    JSON Recovery Layer — AI_Qualifier 4.5
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
        return json.loads(cleaned)
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
                f"⚠ Не удалось извлечь token_ids из факта {getattr(fact, 'id', None)}: {e}"
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
# ⭐ Основная функция квалификации
# ============================================================

def qualify_documents(
    case_id: str,
    docs: List[Dict[str, Any]],
    city: str = "г. Павлодар",
    investigator_fio: str = "Не указан",
    investigator_line: str = "Следователь",
    date_str: Optional[str] = None,
) -> Dict[str, Any]:

    # базовая валидация входных документов
    validate_docs(docs)

    if not date_str:
        date_str = datetime.now().strftime("%d.%m.%Y")

    logger.info(
        f"▶️ QUALIFIER 4.5 (token-json): case_id={case_id}, docs={len(docs)}"
    )

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
    # 3) RAG Router
    # =====================================================================
    router = RAGRouter()
    routed_facts: List[LegalFact] = router.route_for_qualifier(filtered_facts)
    logger.info(f"📙 RAG Router: кандидатов до авто-чистки = {len(routed_facts)}")

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

    logger.info(f"📙 RAG Router: после авто-чистки = {len(routed_facts)}")

    if not routed_facts:
        return _empty_result(
            case_id,
            "После авто-чистки фактов не осталось для квалификации.",
            investigator_fio,
            investigator_line,
        )

    # ============================================================
    # 3.2) Crime Classification (по LegalFact)
    # ============================================================
    cls_input = [f for f in routed_facts if getattr(f, "role", "") != "generic_fact"]
    if not cls_input:
        cls_input = routed_facts

    classification = classify_by_tokens(cls_input)
    logger.info("⚖ Crime classification:\n" + format_classification_debug(classification))

    primary_article = classification.get("primary")
    secondary_articles = classification.get("secondary", [])

    # =====================================================================
    # 4) Подготовка payload фактов для LLM (JSON strict)
    # =====================================================================
    facts_payload: List[Dict[str, Any]] = []
    for f in routed_facts:
        d = f.model_dump()
        # для верификатора: гарантируем поле sources, даже если модель называет его иначе
        if "sources" not in d and "source_refs" in d:
            d["sources"] = d.get("source_refs") or []
        facts_payload.append(d)

    # =====================================================================
    # 5) Вызов LLM для «УСТАНОВИЛ» (P_UST_TOKENS_JSON)
    # =====================================================================
    system_prompt = prompts.P_UST_TOKENS_JSON
    user_payload = {"facts": facts_payload}
    user_prompt = json.dumps(user_payload, ensure_ascii=False, indent=2)

    response = ask_llm(system_prompt, user_prompt)

    # =====================================================================
    # 6) Парсинг JSON-ответа LLM по «УСТАНОВИЛ»
    # =====================================================================
    try:
        if response.startswith("[LLM_ERROR]"):
            raise ValueError("LLM returned error marker")

        parsed = safe_json_loads(response)
        if not parsed:
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

    # ------------------------------------------------------------
    # 6.1. Разбиение «УСТАНОВИЛ» на предложения (для логов)
    # ------------------------------------------------------------
    _sentences_plain = split_into_sentences(ustanovil_text)
    logger.info(f"📘 USTANOVIL: разбиение на предложения = {len(_sentences_plain)}")

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

    # сливаем строгий alignment в основной
    if isinstance(base_alignment, dict):
        alignment = {**base_alignment, **strict_al}
    else:
        alignment = strict_al

    # =====================================================================
    # 7) ПОСТАНОВИЛ — LLM (обычный режим, но JSON-вход)
    # =====================================================================
    post_system = prompts.P_POST
    post_payload = {
        "ustanovil_text": ustanovil_text,
        "primary_article": primary_article,
        "secondary_articles": secondary_articles,
    }
    post_user = json.dumps(post_payload, ensure_ascii=False, indent=2)

    post_text = ask_llm(post_system, post_user)
    if post_text.startswith("[LLM_ERROR]"):
        post_text = _fallback_postanovil(ustanovil_text)

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
        # авто-классификация состава
        "auto_classification": classification,
        "primary_article": primary_article,
        "secondary_articles": secondary_articles,

        # метаданные генерации
        "generation_id": str(uuid.uuid4()),
        "model_version": MODEL_VERSION,
        "case_id": case_id,

        # фактологическая база
        "facts_used": facts_payload,
        "used_tokens": used_tokens,

        # текст постановления
        "established_text": ustanovil_text.strip(),
        "final_postanovlenie": post_text.strip(),

        # проверка
        "verification": verification,

        # предложение → токены
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
        f"✔ QUALIFIER 4.5 завершён: facts={len(facts_payload)}, used={len(used_tokens)}"
    )
    return result


# ============================================================
# 🔧 Fallback «УСТАНОВИЛ» (умный, без мусора)
# ============================================================

def _fallback_ustanovil(facts: List[LegalFact]) -> str:
    """
    Умный fallback: строит краткую юридическую фабулу из фактов,
    без токенов, без мусора, без source_refs.
    Соответствует логике ст. 204 УПК РК (общая фабула).
    """

    persons = set()
    amounts = []
    actions = set()
    dates = set()
    platforms = set()

    for f in facts:
        txt = (getattr(f, "text", "") or "").lower()
        tokens = getattr(f, "tokens", []) or []

        for t in tokens:
            if t.type == "person" and t.value:
                v = t.value.strip()
                if len(v) > 2 and v.lower() not in _BAD_PERSON_TOKENS:
                    persons.add(v)

            if t.type == "amount":
                amounts.append(t.value)

            if t.type == "date":
                dates.add(t.value)

        if any(w in txt for w in ["перевел", "перевела", "отправил", "отправила", "внес", "внесла", "пополнил", "пополнила"]):
            actions.add("переводы денежных средств")

        if any(w in txt for w in ["обман", "обманным путем", "ввел в заблуждение", "ввела в заблуждение", "ввели в заблуждение"]):
            actions.add("введение потерпевших в заблуждение")

        if "usdt" in txt or "okx" in txt or "binance" in txt:
            platforms.add("криптовалютные операции")

    lines: List[str] = []
    lines.append("По материалам дела установлено следующее.")

    if actions:
        lines.append(f"Зафиксированы действия, связанные с {', '.join(sorted(actions))}.")

    if persons:
        lines.append(
            f"В деле фигурируют следующие участники: {', '.join(sorted(persons))}."
        )

    if amounts:
        try:
            # грубая нормализация для min/max: убираем нецифровые
            normalized = []
            for a in amounts:
                digits = re.sub(r"[^\d]", "", a)
                if digits:
                    normalized.append(int(digits))
            if normalized:
                min_v = min(normalized)
                max_v = max(normalized)
                lines.append(
                    f"Отмечены операции на значительные суммы, ориентировочно от {min_v} до {max_v} тенге."
                )
        except Exception:
            # если не смогли нормализовать — просто перечислим суммы
            lines.append(
                f"Отмечены операции на следующие суммы: {', '.join(amounts)}."
            )

    if platforms:
        lines.append("Имеются сведения об операциях, связанных с криптовалютными платформами.")

    if dates:
        lines.append(f"События относятся к датам: {', '.join(sorted(dates))}.")

    lines.append(
        "Указанные обстоятельства в совокупности свидетельствуют о совершении действий имущественного характера с использованием введения в заблуждение и привлечения денежных средств потерпевших."
    )

    return " ".join(lines).strip()


def _fallback_postanovil(ustanovil_text: str) -> str:
    return (
        "ПОСТАНОВИЛ:\n"
        "На основании изложенного в разделе «УСТАНОВИЛ»,\n"
        "требуется получение дополнительных данных для окончательной квалификации.\n"
    )


# ============================================================
# 🔧 Пустой результат
# ============================================================

def _empty_result(case_id: str, msg: str, fio: str, line: str) -> Dict[str, Any]:
    return {
        "generation_id": None,
        "model_version": MODEL_VERSION,
        "case_id": case_id,
        "established_text": msg,
        "final_postanovlenie": msg,
        "facts_used": [],
        "used_tokens": [],
        "verification": {"error": msg, "overall_ok": False},
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "investigator_fio": fio,
        "investigator_line": line,
        "city": None,
        "date": None,
    }
