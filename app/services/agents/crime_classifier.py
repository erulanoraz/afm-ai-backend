# Crime Classifier v6.0 UNIVERSAL
# Равномерная, честная классификация по фактам, без приоритета конкретных статей.

from typing import List, Dict, Any, Optional
from app.services.facts.fact_models import LegalFact, FactToken
from app.services.agents.ai_laws import ALL_AFM_LAWS


# ============================================================
# Доступные статьи
# ============================================================

ARTICLE_CANDIDATES = [
    "189", "190", "214", "216", "217", "218", "301-1"
]

VALID_ARTICLES = [a for a in ARTICLE_CANDIDATES if a in ALL_AFM_LAWS]


# ============================================================
# Ключевые сигналы — мягкие и равномерные
# ============================================================

ARTICLE_KEYWORDS: Dict[str, Dict[str, list[str]]] = {

    # Мошенничество
    "190": {
        "core": ["обман", "ввел в заблуждение", "заблужд", "ложн"],
        "context": ["перевел", "отправил", "получил", "деньги"],
    },

    # Присвоение/Растрата
    "189": {
        "core": ["присвоил", "присвоила", "вверен", "растрата"],
        "context": ["имущество", "доверено"],
    },

    # Незаконная предпринимательская деятельность
    "214": {
        "core": ["незаконная предприним", "без регистрации", "без лиценз"],
        "context": ["доход", "товары", "деятельность"],
    },

    # Лжепредпринимательство
    "216": {
        "core": ["фиктив", "лжепредприят", "подставн"],
        "context": ["счет-фактура", "обналич"],
    },

    # Финансовая пирамида
    "217": {
        "core": ["финансовая пирамида", "пирамида", "инвестиц", "рефераль"],
        "context": ["вклад", "проценты", "дивиденды", "вложил"],
    },

    # Легализация доходов
    "218": {
        "core": ["легализац", "отмыван", "скрыть происхождение"],
        "context": ["финансовый поток", "перевод средств", "подставные"],
    },

    # Вейпы / электронные сигареты
    "301-1": {
        "core": ["вейп", "электронн сигарет", "никотин"],
        "context": ["реализация", "продажа"],
    },
}



# ============================================================
# Вспомогательные функции
# ============================================================

def _text(fact: LegalFact) -> str:
    tokens = " ".join((t.value or "").lower() for t in fact.tokens)
    return (fact.text or "").lower() + " " + tokens


def _has_amount(f: LegalFact) -> bool:
    return any(t.type == "amount" for t in f.tokens)


def _has_transfer_tokens(f: LegalFact) -> bool:
    return any(t.type in ("digital_transfer", "account", "channel") for t in f.tokens)


def _safe_lower(s: Optional[str]) -> str:
    return (s or "").lower()



# ============================================================
# Базовое взвешивание признаков
# ============================================================

def _score_article(article: str, fact: LegalFact) -> Dict[str, Any]:
    text = _text(fact)

    score = 0.0
    reasons = []

    cfg = ARTICLE_KEYWORDS.get(article, {})
    core_kw = cfg.get("core", [])
    ctx_kw = cfg.get("context", [])

    # --- Основные сигналы ---
    for w in core_kw:
        if w in text:
            score += 1.6
            reasons.append(f"core keyword: {w}")

    # --- Контекст ---
    for w in ctx_kw:
        if w in text:
            score += 0.7
            reasons.append(f"context keyword: {w}")

    # --- Суммы ---
    if _has_amount(fact):
        score += 0.6
        reasons.append("amount: деньги")

    # --- Переводы ---
    if _has_transfer_tokens(fact):
        score += 0.5
        reasons.append("transfer: перевод средств")

    return {"score": score, "reasons": reasons}



# ============================================================
# Главная функция классификации
# ============================================================

def classify_by_tokens(facts: List[LegalFact]) -> Dict[str, Any]:

    result = {
        "primary": None,
        "secondary": [],
        "scores": {}
    }

    if not facts:
        return result

    # Суммарные баллы
    scores = {a: 0.0 for a in VALID_ARTICLES}
    reasons_map = {a: [] for a in VALID_ARTICLES}

    # Проходим по фактам
    for f_idx, fact in enumerate(facts, start=1):
        fact_id = getattr(fact, "fact_id", f"fact_{f_idx}")

        for art in VALID_ARTICLES:
            sc = _score_article(art, fact)
            if sc["score"] > 0:
                scores[art] += sc["score"]
                for r in sc["reasons"]:
                    reasons_map[art].append(f"[{fact_id}] {r}")

    # Сохраняем
    for art in VALID_ARTICLES:
        result["scores"][art] = {
            "score": round(scores[art], 3),
            "reasons": reasons_map[art],
        }

    # Пороги мягкие и честные
    THRESH_PRIMARY = 3.0
    THRESH_SECONDARY = 1.8

    # Primary — статья с максимальным score
    primary = max(scores, key=lambda a: scores[a])
    if scores[primary] >= THRESH_PRIMARY:
        result["primary"] = primary
    else:
        result["primary"] = None

    # Secondary — все статьи, которые имеют вес
    secondary = [
        art for art, sc in scores.items()
        if art != primary and sc >= THRESH_SECONDARY
    ]
    result["secondary"] = secondary

    return result



# ============================================================
# Форматированный вывод для логов
# ============================================================

def format_classification_debug(classification: Dict[str, Any]) -> str:

    lines = []

    primary = classification.get("primary")
    secondaries = classification.get("secondary", [])

    lines.append(f"PRIMARY: {primary or 'не определена'}")

    if secondaries:
        lines.append("SECONDARY: " + ", ".join(secondaries))
    else:
        lines.append("SECONDARY: —")

    for art, d in classification["scores"].items():
        if d["score"] > 0:
            lines.append(f"\nСтатья {art}: score={d['score']}")
            for r in d["reasons"][:6]:
                lines.append(f"  • {r}")

    return "\n".join(lines)
