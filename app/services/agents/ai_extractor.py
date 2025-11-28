# app/services/agents/ai_extractor.py
"""
AI Extractor 3.1 — улучшенная версия под ChatGPT-style RAG токенизацию.

Главные задачи:
    1. Очистить текст (SUPER PRE-FILTER 3.1)
    2. Разбить на предложения
    3. Применить FactTokenizer → получить LegalFact
    4. Собрать факты через FactGraph (слияние дубликатов)
    5. Вернуть структурированные LegalFact

Особенности:
    • аккуратная фильтрация процессуального мусора
    • минимальное вмешательство в фабулу
    • чистые токены (amount, date, person, action, …)
    • строгая структура LegalFact
"""

import logging
import re
from typing import List

from app.services.facts.fact_models import LegalFact
from app.services.facts.fact_tokenizer import FactTokenizer
from app.services.facts.fact_graph import FactGraph

logger = logging.getLogger(__name__)


# =====================================================================
# 🔥 SUPER PRE-FILTER 3.1 – аккуратное удаление процессуального мусора
# =====================================================================

DIALOG_PATTERNS = [
    r"вопрос[:\s].*",
    r"ответ[:\s].*",
    r"вопрос следовател[яй].*",
]

TECH_MATERIAL = [
    r"допрос окончен.*",
    r"приложени[ея].*",
    r"протокол допроса.*",
    r"ордер №.*",
    r"дата печати.*",
    r"просмотрено.*",
    r"электронный документ.*",
    r"подпись наложена.*",
    r"пояснил.*",
    r"объяснил.*",
]

PERSON_FORM = [
    r"фамилия[:\s].*",
    r"имя[:\s].*",
    r"отчество[:\s].*",
    r"место рождения.*",
    r"место жительства.*",
    r"дата рождения.*",
    r"национальност.*",
    r"гражданств.*",
]

def super_pre_filter(text: str) -> str:
    """
    Удаляет процессуальный мусор, но НЕ трогает фабулу.
    Возвращает очищенный текст.
    """
    t = text.strip()
    if not t:
        return ""

    # Сначала удаляем крупные блоки
    for p in DIALOG_PATTERNS + TECH_MATERIAL + PERSON_FORM:
        t = re.sub(p, "", t, flags=re.IGNORECASE)

    # Убираем длинные повторяющиеся пробелы
    t = re.sub(r"\s+", " ", t, flags=re.IGNORECASE)

    return t.strip()


# =====================================================================
# 🔍 Минимальное разбиение на предложения
# =====================================================================

def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 5]


# =====================================================================
# 🧠 ГЛАВНАЯ ФУНКЦИЯ EXTRACT_ALL — НОВАЯ АРХИТЕКТУРА
# =====================================================================

def extract_all(docs: List[dict]) -> List[LegalFact]:
    """
    Вход:
        docs = [{ file_id, page, text }]
    Выход:
        List[LegalFact]
    """

    if not docs:
        logger.warning("⚠ extract_all: docs пусты")
        return []

    tokenizer = FactTokenizer()
    graph = FactGraph()

    cleaned_docs = []

    # ---------------------------------------------------------
    # 1) PRE-FILTER
    # ---------------------------------------------------------
    for d in docs:
        file_id = d.get("file_id")
        page = d.get("page", 1)
        text = d.get("text", "") or ""

        cleaned = super_pre_filter(text)

        if not cleaned or len(cleaned) < 5:
            continue

        cleaned_docs.append({
            "file_id": file_id,
            "page": page,
            "text": cleaned,
        })

    if not cleaned_docs:
        logger.warning("⚠ extract_all: после pre-filter ничего не осталось")
        return []

    # ---------------------------------------------------------
    # 2) TOKENIZATION
    # ---------------------------------------------------------
    logger.info(f"🟦 FactTokenizer: вход документов = {len(cleaned_docs)}")

    tokenized_facts = tokenizer.tokenize(cleaned_docs)

    logger.info(f"🟩 FactTokenizer: извлечено LegalFacts = {len(tokenized_facts)}")

    if not tokenized_facts:
        return []

    # ---------------------------------------------------------
    # 3) FACT GRAPH MERGE
    # ---------------------------------------------------------
    merged_facts = graph.build(tokenized_facts)

    logger.info(f"🟧 FactGraph: после объединения = {len(merged_facts)} фактов")

    return merged_facts
