# app/services/reranker.py
import re
import logging
from typing import List, Dict, Any

from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ============================================================
# 🔥 Мягкая очистка текста (ничего важного не удаляем)
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
    ]
    for g in garbage:
        t = re.sub(g, "", t, flags=re.IGNORECASE)

    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


# ============================================================
# 🔥 RERANKER PRO 3.0 — усиливаем ключевые документы
# ============================================================

class LLMReranker:
    """
    Reranker PRO 3.0:
    • LLM cross-encoder
    • Жёсткий приоритет для:
        - рапорта
        - рапорт ЕРДР
        - постановлений
        - протокола допроса подозреваемого
        - протокола допроса потерпевшего
    """

    def __init__(self):
        self.llm = LLMClient()

    def rerank(self, query: str, items: List[Dict[str, Any]], top_k: int = 75) -> List[Dict[str, Any]]:
        if not items:
            return []

        # =======================
        # 1. очистка текста
        # =======================
        cleaned_items: List[Dict[str, Any]] = []
        for it in items:
            cleaned = clean_text(it.get("text", ""))
            if not cleaned:
                continue
            n = dict(it)
            n["clean_text"] = cleaned
            cleaned_items.append(n)

        if not cleaned_items:
            logger.warning("⚠ Reranker: после очистки нет текстов")
            return []

        # =======================
        # 2. нарезка фрагментов
        # =======================
        snippets = [
            f"{i+1}. {it['clean_text'][:500]}"
            for i, it in enumerate(cleaned_items)
        ]

        prompt = f"""
Ты — модель ранжирования. Оцени релевантность каждого фрагмента запросу
по шкале 0.0–1.0.

Ответ: только JSON массив чисел, например:
[0.91, 0.12, 0.44]

Запрос:
"{query}"

Фрагменты:
{chr(10).join(snippets)}
"""

        # =======================
        # 3. LLM вызов
        # =======================
        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}])
        except Exception as e:
            logger.error(f"❌ Reranker LLM error: {e}")
            return cleaned_items[:min(top_k, len(cleaned_items))]

        resp_text = str(resp)
        arr = re.findall(r"[-+]?\d*\.\d+|\d+", resp_text)
        scores = [float(x) for x in arr[:len(cleaned_items)]] if arr else []

        if len(scores) != len(cleaned_items):
            logger.warning(f"⚠ Reranker: mismatch scores {len(scores)} vs {len(cleaned_items)}")
            while len(scores) < len(cleaned_items):
                scores.append(0.0)

        # назначаем базовые оценки
        for it, s in zip(cleaned_items, scores):
            it["cross_score"] = float(s)

        # ============================================================
        # 4. ЖЁСТКИЙ приоритет для ключевых документов
        # ============================================================

        STRONG_BOOST = 0.99999
        MEDIUM_BOOST = 0.97
        WEAK_BOOST = 0.85

        def lower_filename(it):
            return (it.get("filename") or "").lower()

        strong_filename_markers = [
            "протокол_допроса_подозреваемого",
            "протокол_допроса_подозреваемой",
            "рапорт_куи",
            "ердр",
            "постановление_о_признании_лица_потерпевшим",
        ]

        medium_filename_markers = [
            "протокол_допроса_потерпевшего",
            "постановление_о_признании_лица_гражданским_истцом",
            "постановление_о_возбуждении",
        ]

        # сильные текстовые маркеры
        strong_text_markers = [
            "протокол допроса подозреваемого",
            "протокол допроса подозреваемой",
            "он подозревается",
            "она подозревается",
            "сообщено о подозрении",
        ]

        # мягкие
        soft_text_markers = [
            "допрос потерпевшего",
            "в качестве подозреваемого",
            "в отношении",
            "гражданин",
            "гражданка",
        ]

        # ПРОХОД 1 → УСТАНАВЛИВАЕМ ЖЁСТКИЕ ПРИОРИТЕТЫ
        for it in cleaned_items:
            fn = lower_filename(it)
            txt = it["clean_text"].lower()

            # filename — super-priority
            if any(m in fn for m in strong_filename_markers):
                it["cross_score"] = STRONG_BOOST
                continue

            # text — super-priority
            if any(m in txt for m in strong_text_markers):
                it["cross_score"] = STRONG_BOOST
                continue

            # filename — medium
            if any(m in fn for m in medium_filename_markers):
                it["cross_score"] = max(it["cross_score"], MEDIUM_BOOST)

            # text — medium
            if any(m in txt for m in soft_text_markers):
                it["cross_score"] = max(it["cross_score"], WEAK_BOOST)

        # =======================
        # 5. сортировка
        # =======================
        sorted_items = sorted(cleaned_items, key=lambda x: x["cross_score"], reverse=True)

        # =======================
        # 6. возвращаем TOP 75
        # =======================
        return sorted_items[:min(top_k, len(sorted_items))]
