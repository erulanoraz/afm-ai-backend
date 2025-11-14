# app/services/reranker.py
import json
import re
from app.services.llm_client import LLMClient  # корректный импорт, если клиент в app/services/llm_client.py


class LLMReranker:
    """
    Легкий LLM-based reranker: пересортировывает документы по уместности к запросу.
    Использует API модели (например, через local LLM endpoint).
    """

    def __init__(self):
        self.llm = LLMClient()

    def rerank(self, query: str, items):
        """
        query: строка запроса
        items: список документов [{ "text": "...", ... }]
        Возвращает тот же список, но с ключом cross_score и отсортированный.
        """
        if not items:
            return []

        # Формируем сокращённый список текстов для LLM
        snippets = [f"{i+1}. {it['text'][:400]}" for i, it in enumerate(items)]
        prompt = f"""
Задача: оцени релевантность каждого документа запросу от 0 до 1.
Ответ должен быть только в виде JSON-массива чисел, без текста.

Запрос:
{query}

Документы:
{chr(10).join(snippets)}
"""

        # Отправляем запрос к LLM
        resp = self.llm.chat([{"role": "user", "content": prompt}])

        # Пробуем извлечь числа (0–1) из текста
        arr = re.findall(r"[-+]?\d*\.\d+|\d+", resp)
        scores = [float(x) for x in arr[:len(items)]] if arr else [0.0] * len(items)

        # Присваиваем каждому элементу cross_score
        for it, s in zip(items, scores):
            it["cross_score"] = s

        # Сортировка по убыванию
        return sorted(items, key=lambda x: x.get("cross_score", 0.0), reverse=True)
