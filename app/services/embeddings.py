# app/services/embeddings.py
import logging
from typing import List

from app.search.vector_client import get_vector_client

logger = logging.getLogger("EMBEDDINGS")


def embed_text(text: str) -> List[float]:
    """
    Пытается получить embedding для текста через Weaviate, используя тот же
    vectorizer (text2vec-transformers), что и для чанков.

    Технически Weaviate не умеет "просто векторизовать текст" как OpenAI API,
    поэтому мы используем near_text-поиск и берём вектор ближайшего чанка.

    ВАЖНО:
    - Это работает корректно только если индекс Chunk НЕ пустой
      (ингест реально кладёт чанки в Weaviate).
    - Если индекс пуст, функция вернёт [] и это надо воспринимать
      как отсутствие семантического слоя.
    """
    text = (text or "").strip()
    if not text:
        return []

    try:
        vc = get_vector_client()

        result = vc.search(query_text=text, limit=1, with_vector=True)
        # Ожидаем структуру вида:
        # {"data": {"Get": {"Chunk": [ { "_additional": {"vector": [...]} , ... } ] } } }
        hits = (
            result.get("data", {})
            .get("Get", {})
            .get("Chunk", [])
        )

        if not hits:
            logger.warning(
                "[embed_text] Weaviate вернул 0 объектов. "
                "Скорее всего, класс Chunk пуст или не проиндексирован."
            )
            return []

        first = hits[0]
        additional = first.get("_additional", {}) or {}
        vec = additional.get("vector") or []

        if not vec:
            logger.error(
                "[embed_text] _additional.vector отсутствует в ответе Weaviate. "
                "Проверь конфигурацию модуля text2vec-transformers и схему Chunk."
            )

        return vec

    except Exception as e:
        logger.error(f"[embed_text ERROR] {e}")
        return []


def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Батч-обёртка над embed_text.
    НЕ оптимальна по производительности (делает N запросов), но проста
    и достаточна для отладки / небольших объёмов.
    """
    vectors: List[List[float]] = []

    for t in texts:
        try:
            vec = embed_text(t)
            vectors.append(vec)
        except Exception as e:
            logger.error(f"[embed_batch ERROR] text='{t[:30]}...' : {e}")
            vectors.append([])

    return vectors
