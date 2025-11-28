# app/services/agents/ai_completeness.py
"""
AI Completeness 4.0 — модуль проверки достаточности фактов
для построения раздела «УСТАНОВИЛ» и последующей квалификации.

Работает ТОЛЬКО на FactToken.

Каждый FactToken имеет поля:
    token_id: str
    text: str
    type: event | status | context | meta
    role: Optional[str]
    action: Optional[str]
    date: Optional[str]
    amount: Optional[str]
    sources: List[FactSource]
    confidence: float
"""

from typing import List, Dict, Any
from app.services.facts.fact_models import LegalFact


def analyze_completeness(tokens: List[LegalFact]) -> Dict[str, Any]:
    """
    Анализирует полноту фактологической базы.
    Возвращает структуру вида:

    {
        "has_dates": True,
        "has_roles": True,
        "has_actions": True,
        "has_amounts": False,
        "has_sources": True,
        "has_events": True,
        "missing": ["amounts"],
        "score": 0.78
    }
    """

    if not tokens:
        return {
            "has_dates": False,
            "has_roles": False,
            "has_actions": False,
            "has_amounts": False,
            "has_sources": False,
            "has_events": False,
            "missing": ["all"],
            "score": 0.0
        }

    has_dates = False
    has_roles = False
    has_actions = False
    has_amounts = False
    has_sources = False
    has_events = False

    for t in tokens:
        if t.date:
            has_dates = True
        if t.role:
            has_roles = True
        if t.action:
            has_actions = True
        if t.amount:
            has_amounts = True
        if t.source_refs and len(t.source_refs) > 0:
            has_sources = True
        if t.type == "event":
            has_events = True

    missing = []
    if not has_events:
        missing.append("events")
    if not has_dates:
        missing.append("dates")
    if not has_roles:
        missing.append("roles")
    if not has_actions:
        missing.append("actions")
    if not has_amounts:
        missing.append("amounts")
    if not has_sources:
        missing.append("sources")

    # Простейший completeness-score (0—1)
    completeness_score = (
        (1 if has_events else 0)
        + (1 if has_dates else 0)
        + (1 if has_roles else 0)
        + (1 if has_actions else 0)
        + (1 if has_amounts else 0)
        + (1 if has_sources else 0)
    ) / 6.0

    return {
        "has_dates": has_dates,
        "has_roles": has_roles,
        "has_actions": has_actions,
        "has_amounts": has_amounts,
        "has_sources": has_sources,
        "has_events": has_events,
        "missing": missing,
        "score": round(completeness_score, 3)
    }


# ================================================================
# Удобный вспомогательный метод — форматирование для LLM
# ================================================================

def summarize_missing_to_text(completeness: Dict[str, Any]) -> str:
    """
    Преобразует список отсутствующих элементов в удобный текст.
    Например: "отсутствуют даты, суммы и роли"
    """

    missing = completeness.get("missing", [])
    if not missing:
        return "все ключевые данные присутствуют"

    names = {
        "events": "описание событий",
        "dates": "даты",
        "roles": "роли участников",
        "actions": "действия",
        "amounts": "суммы",
        "sources": "источники"
    }

    readable = [names.get(m, m) for m in missing]

    return "отсутствуют: " + ", ".join(readable)
