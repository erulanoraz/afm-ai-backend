# app/services/reranker.py
import re
import json
import logging
from typing import List, Dict, Any

from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

TOP_RERANK_OUTPUT = 50  # ← было 120, теперь меньше но качественнее


class LLMReranker:
    def __init__(self):
        self.llm = LLMClient()

    # ============================================================
    # 🔧 Локальный baseline
    # ============================================================
    def _compute_baseline_score(self, doc: Dict[str, Any]) -> float:
        """Простой baseline по типу файла"""
        fn = (doc.get("filename") or "").lower()
        txt = (doc.get("text") or "").lower()

        score = 0.0

        # Протоколы допроса подозреваемого — ГЛАВНОЕ
        if any(m in fn for m in ["протокол_допроса_подозреваем", "допроса подозреваем", "куи"]):
            score += 3.0

        # Рапорты, ердр
        if any(m in fn for m in ["рапорт", "ердр"]):
            score += 2.5

        # Допросы потерпевших
        if any(m in fn for m in ["допроса_потерпевш", "допроса потерпевш"]):
            score += 1.5

        # Постановления
        if "постановление" in fn:
            score += 1.0

        # Ключевые слова в тексте
        if any(k in txt for k in ["он подозревается", "она подозревается", "совершил"]):
            score += 0.8

        if any(k in txt for k in ["перевел", "получил", "внес", "вложил"]):
            score += 0.6

        if any(k in txt for k in ["тенге", "тг", "денежные средства", "ущерб"]):
            score += 0.5

        return score

    # ============================================================
    # 🔥 ИСПРАВЛЕННЫЙ rerank с ROBUST JSON парсингом
    # ============================================================
    def rerank(self, query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ранжирование с LLM + падбэком на baseline.
        ГЛАВНОЕ: robust JSON парсинг!
        """
        if not items:
            return []

        cleaned_items: List[Dict[str, Any]] = []
        
        for it in items:
            text = it.get("text") or ""
            if len(text) < 20:
                continue
            
            doc = dict(it)
            doc["clean_text"] = text[:500]  # ограничиваем длину
            cleaned_items.append(doc)

        if not cleaned_items:
            return []

        # 1) Baseline score
        for d in cleaned_items:
            d["baseline_score"] = self._compute_baseline_score(d)

        # 2) Подготовка для LLM (КОРОЧЕ!)
        snippets: List[str] = []
        for i, doc in enumerate(cleaned_items):
            prefix = f"[{i}] {doc.get('filename', '')} стр.{doc.get('page', '?')}: "
            body = doc["clean_text"][:300]
            snippets.append(prefix + body)

        # 3) ПРОСТОЙ prompt (без сложности)
        system_prompt = (
            "Ты — оцениватель релевантности для уголовного дела. "
            "Оцени каждый документ от 0.0 до 1.0. "
            "0.0 = неважно, 1.0 = очень важно."
        )

        user_prompt = f"""
Оцени релевантность документов. Верни ТОЛЬКО JSON массив чисел: [0.8, 0.3, 0.9, ...]

Ищем: факты преступления, переводы денег, обман, действия подозреваемого.

Документы:
{chr(10).join(snippets[:20])}

Ответ (только JSON):
"""

        # 4) LLM scoring с ROBUST парсингом
        llm_scores = [0.0] * len(cleaned_items)

        try:
            resp = self.llm.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])

            # Вариант 1: чистый JSON массив
            try:
                parsed = json.loads(resp.strip())
                if isinstance(parsed, list):
                    for i, val in enumerate(parsed[:len(cleaned_items)]):
                        if isinstance(val, (int, float)):
                            llm_scores[i] = float(val)
            except json.JSONDecodeError:
                pass

            # Вариант 2: JSON в строке
            if llm_scores == [0.0] * len(cleaned_items):
                match = re.search(r"\[[\d\.,\s]+\]", resp)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                        if isinstance(parsed, list):
                            for i, val in enumerate(parsed[:len(cleaned_items)]):
                                if isinstance(val, (int, float)):
                                    llm_scores[i] = float(val)
                    except json.JSONDecodeError:
                        pass

            # Вариант 3: Regex вытаскиваем числа
            if llm_scores == [0.0] * len(cleaned_items):
                nums = re.findall(r"0?\.\d+", resp)
                for i, num_str in enumerate(nums[:len(cleaned_items)]):
                    try:
                        llm_scores[i] = float(num_str)
                    except ValueError:
                        pass

            logger.info(f"✅ Reranker: LLM оценки = {llm_scores}")

        except Exception as e:
            logger.error(f"⚠️ LLM error: {e}, используем baseline")

        # 5) Комбинируем baseline + LLM
        for d, llm_s in zip(cleaned_items, llm_scores):
            baseline_s = float(d["baseline_score"]) / 4.0  # нормализуем
            d["llm_score"] = float(llm_s)
            d["cross_score"] = baseline_s * 0.4 + llm_s * 0.6

        # 6) Сортировка
        sorted_items = sorted(cleaned_items, key=lambda d: d["cross_score"], reverse=True)

        logger.info(f"📊 Reranker output: {len(sorted_items[:TOP_RERANK_OUTPUT])} документов")
        return sorted_items[:TOP_RERANK_OUTPUT]