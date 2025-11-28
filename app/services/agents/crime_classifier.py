"""
Crime Classifier 4.0 — классификация состава преступления по LegalFact (FactToken групповая модель)
Работает ТОЛЬКО по LegalFact, где каждый факт состоит из набора FactToken.
"""

from typing import List, Dict, Any, Optional
from app.services.facts.fact_models import LegalFact, FactToken
from app.services.agents.ai_laws import ALL_AFM_LAWS


# ============================================================
# 🔹 Кандидаты статей
# ============================================================

ARTICLE_CANDIDATES = [
    "189",
    "190",
    "214",
    "216",
    "217",
    "218",
    "301-1",
]

VALID_ARTICLES = [a for a in ARTICLE_CANDIDATES if a in ALL_AFM_LAWS]


# ============================================================
# 🔹 Ключевые слова
# ============================================================

ARTICLE_KEYWORDS: Dict[str, Dict[str, list[str]]] = {
    "190": {
        "core": ["мошеннич", "обман", "ввел в заблужден", "заблужден", "ложн"],
        "context": ["интернет", "онлайн", "платформ", "сайт"],
    },
    "189": {
        "core": ["вверен", "растрата", "присво", "подотчет", "материально ответ"],
        "context": ["имущество было передано"],
    },
    "214": {
        "core": ["без регистрации", "без лицензии", "незаконная предприним"],
        "context": ["получение дохода", "подакциз"],
    },
    "216": {
        "core": ["счет-фактур", "фиктив", "без фактического"],
        "context": ["обналич", "наличн"],
    },
    "217": {
        "core": ["финансовая пирамида", "инвестиционная пирамида", "пирамида"],
        "context": ["вклад", "вложен", "инвестиц"],
    },
    "218": {
        "core": ["легализац", "отмыван", "скрыть происхождение"],
        "context": ["подставные лица", "финансовый поток"],
    },
    "301-1": {
        "core": ["вейп", "электронн сигарет", "некурительн табач"],
        "context": ["продажа", "оптовая партия"],
    },
}


# ============================================================
# 🔹 Утилиты
# ============================================================

def _safe_lower(x: Optional[str]) -> str:
    return (x or "").lower()


def _fact_text(fact: LegalFact) -> str:
    """
    Создаёт текст факта — объединённый текст всех FactToken.value
    """
    return " ".join(t.value for t in fact.tokens if t.value).lower()


def _get_amounts(fact: LegalFact) -> List[str]:
    return [t.value for t in fact.tokens if t.type == "amount"]


def _get_actions(fact: LegalFact) -> List[str]:
    return [t.value for t in fact.tokens if t.type == "action"]


def _score_article_for_token(article_id: str, fact: LegalFact) -> Dict[str, Any]:
    """
    Считает score для ОДНОГО LegalFact по ОДНОЙ статье.
    """
    text = _fact_text(fact)

    keywords = ARTICLE_KEYWORDS.get(article_id, {})
    core_kws = keywords.get("core", [])
    ctx_kws = keywords.get("context", [])

    score = 0.0
    reasons: list[str] = []

    # 1) Ключевые слова
    for kw in core_kws:
        if kw in text:
            score += 1.5
            reasons.append(f"core_keyword: {kw}")

    for kw in ctx_kws:
        if kw in text:
            score += 0.5
            reasons.append(f"context_keyword: {kw}")

    # 2) Суммы усиливают экономические статьи
    if _get_amounts(fact) and article_id in ["189", "190", "214", "216", "217", "218", "301-1"]:
        score += 0.5
        reasons.append("amount: есть сумма")

    # 3) Действия (пример для мошенничества)
    actions = _get_actions(fact)
    if article_id == "190":
        if any("обман" in _safe_lower(a) for a in actions):
            score += 1.0
            reasons.append("action: признаки обмана")

    # 4) Роль факта
    if fact.role:
        r = fact.role.lower()
        if article_id == "190" and "suspect" in r:
            score += 0.5
            reasons.append("role: подозреваемый")
        if article_id == "189" and "respons" in r:
            score += 1.0
            reasons.append("role: ответственное лицо")

    return {"score": score, "reasons": reasons}


# ============================================================
# 🔹 Главная функция классификации
# ============================================================

def classify_by_tokens(facts: List[LegalFact]) -> Dict[str, Any]:
    """
    Вход:
        facts: List[LegalFact]

    Выход:
        {
            "primary": "190" | "217" | ... | None,
            "secondary": ["214", "218"],
            "scores": {
                "190": {"score": 5.0, "reasons": [...]},
                ...
            }
        }
    """
    result: Dict[str, Any] = {
        "primary": None,
        "secondary": [],
        "scores": {},
    }

    if not facts:
        return result

    scores: Dict[str, float] = {a: 0.0 for a in VALID_ARTICLES}
    reasons_map: Dict[str, List[str]] = {a: [] for a in VALID_ARTICLES}

    # Обрабатываем каждый факт
    for idx, f in enumerate(facts, start=1):
        # красивый id для логов: либо fact_id, либо fact_N
        fact_label = getattr(f, "fact_id", None) or f"fact_{idx}"

        for art in VALID_ARTICLES:
            res = _score_article_for_token(art, f)
            if res["score"] > 0:
                scores[art] += res["score"]
                reasons_map[art].extend(
                    [f"[{fact_label}] {msg}" for msg in res["reasons"]]
                )

    # Сохраняем score по статьям
    for art in VALID_ARTICLES:
        result["scores"][art] = {
            "score": round(scores[art], 3),
            "reasons": reasons_map[art],
        }

    # Пороги
    THRESH_PRIMARY = 3.0
    THRESH_SECONDARY = 2.0

    # Primary — максимальный score
    primary: Optional[str] = None
    max_score = 0.0
    for art, sc in scores.items():
        if sc > max_score:
            max_score = sc
            primary = art

    if primary and max_score >= THRESH_PRIMARY:
        result["primary"] = primary
    else:
        primary = None

    # Secondary — все, кто ≥ THRESH_SECONDARY и не primary
    secondary: list[str] = []
    for art, sc in scores.items():
        if art == primary:
            continue
        if sc >= THRESH_SECONDARY:
            secondary.append(art)

    result["secondary"] = secondary

    return result


# ============================================================
# 🔹 Форматирование для логов
# ============================================================

def format_classification_debug(classification: Dict[str, Any]) -> str:
    lines: List[str] = []

    primary = classification.get("primary")
    secondary = classification.get("secondary", [])
    scores = classification.get("scores", {})

    lines.append(f"PRIMARY: {primary or 'не определена'}")
    if secondary:
        lines.append(f"SECONDARY: {', '.join(secondary)}")
    else:
        lines.append("SECONDARY: —")

    for art, data in scores.items():
        sc = data.get("score", 0.0)
        if sc <= 0:
            continue
        lines.append(f"\nСтатья {art}: score={sc}")
        for r in data.get("reasons", [])[:5]:
            lines.append(f"  • {r}")

    return "\n".join(lines)
