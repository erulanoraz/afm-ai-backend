# app/services/reranker.py
import re
import logging
from typing import List, Dict, Any

from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ============================================================
# 🧹 Мягкая очистка текста (ничего важного не удаляем)
# ============================================================

def clean_text(text: str) -> str:
    if not text:
        return ""

    t = text.strip()

    garbage = [
        r"©\s?Все права защищены",
        r"сканировано\s?с\s?помощью.*",
        r"страница\s?\d+\s?из\s?\d+",
        r"QR[- ]?код.*",
        r"Документ создан.*",
        r"электронный документ.*",
        r"Просмотрено на.*",
        r"Дата печати.*",
    ]
    for g in garbage:
        t = re.sub(g, "", t, flags=re.IGNORECASE)

    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


# ============================================================
# 🔥 RERANKER PRO 4.1 — baseline + LLM, без final_score
# ============================================================

class LLMReranker:
    """
    Reranker PRO 4.1:
    • baseline приоритет по типу документа
    • LLM-оценка релевантности (0–1)
    • итоговый скор: cross_score = baseline + llm_score
    """

    def __init__(self):
        self.llm = LLMClient()

    def _compute_baseline_score(self, doc: Dict[str, Any]) -> float:
        fn = (doc.get("filename") or "").lower()
        txt = (doc.get("clean_text") or "").lower()

        score = 0.0

        # 🔹 Файловые маркеры
        strong_filename_markers = [
            "протокол_допроса_подозреваемого",
            "протокол_допроса_подозреваемой",
            "рапорт_куи",
            "ердр",
            "рапорт_о_регистрации",
        ]
        medium_filename_markers = [
            "протокол_допроса_потерпевшего",
            "постановление_о_признании_лица_потерпевшим",
            "постановление_о_признании_лица_гражданским_истцом",
            "постановление_о_возбуждении",
        ]

        # 🔹 Текстовые маркеры
        strong_text_markers = [
            "протокол допроса подозреваемого",
            "протокол допроса подозреваемой",
            "сообщено о подозрении",
            "он подозревается",
            "она подозревается",
            "в качестве подозреваемого",
        ]
        soft_text_markers = [
            "допрос потерпевшего",
            "допрос свидетель",
            "потерпевший пояснил",
            "потерпевшая пояснила",
        ]

        if any(m in fn for m in strong_filename_markers):
            score += 2.0
        if any(m in fn for m in medium_filename_markers):
            score += 1.0

        if any(m in txt for m in strong_text_markers):
            score += 2.0
        if any(m in txt for m in soft_text_markers):
            score += 0.5

        return score

    def rerank(self, query: str, items: List[Dict[str, Any]], top_k: int = 75) -> List[Dict[str, Any]]:
        if not items:
            return []

        # 1️⃣ Очистка текста
        cleaned_items: List[Dict[str, Any]] = []
        for it in items:
            cleaned = clean_text(it.get("text", "") or "")
            if not cleaned:
                continue
            n = dict(it)
            n["clean_text"] = cleaned
            cleaned_items.append(n)

        if not cleaned_items:
            logger.warning("⚠ Reranker: после очистки нет текстов")
            return []

        # 2️⃣ baseline-оценка (без LLM)
        for doc in cleaned_items:
            doc["baseline_score"] = self._compute_baseline_score(doc)

        # 3️⃣ LLM-оценка (мягкая, с fallback)
        snippets = [
            f"{i+1}. {doc['clean_text'][:500]}"
            for i, doc in enumerate(cleaned_items)
        ]

        prompt = f"""
Ты — модель ранжирования. Оцени релевантность каждого фрагмента
к запросу по шкале от 0.0 до 1.0.

Верни ТОЛЬКО JSON-массив чисел, например:
[0.91, 0.12, 0.44]

Запрос:
"{query}"

Фрагменты:
{chr(10).join(snippets)}
"""

        llm_scores = [0.0] * len(cleaned_items)

        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}])
            resp_text = str(resp)
            numbers = re.findall(r"-?\d+(?:\.\d+)?", resp_text)

            for i, num in enumerate(numbers[:len(cleaned_items)]):
                try:
                    llm_scores[i] = float(num)
                except Exception:
                    continue

            logger.info(f"Reranker LLM: получили {len(numbers)} чисел для {len(cleaned_items)} фрагментов")

        except Exception as e:
            logger.error(f"❌ Reranker LLM error, работаем только на baseline: {e}")
            # llm_scores остаются по 0.0

        # 4️⃣ Итоговый скор: cross_score = baseline + llm_score
        for doc, llm_s in zip(cleaned_items, llm_scores):
            doc["llm_score"] = float(llm_s)
            doc["cross_score"] = float(doc.get("baseline_score", 0.0)) + float(llm_s)

        # 5️⃣ Сортировка по cross_score
        sorted_items = sorted(
            cleaned_items,
            key=lambda d: d.get("cross_score", 0.0),
            reverse=True,
        )

        # 6️⃣ Возвращаем TOP K
        return sorted_items[:min(top_k, len(sorted_items))]
