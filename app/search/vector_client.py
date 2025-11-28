# app/services/vector_client.py 5.0 (Weaviate 2.x Evidence Engine)

import logging
from typing import Any, Dict, List, Optional

import weaviate
from weaviate.exceptions import WeaviateBaseError

from app.utils.config import settings

logger = logging.getLogger("VECTOR-CLIENT")

_vector_client_singleton = None


def get_vector_client() -> "VectorClient":
    global _vector_client_singleton
    if _vector_client_singleton is None:
        _vector_client_singleton = VectorClient(settings.WEAVIATE_URL)
    return _vector_client_singleton


class VectorClient:
    """
    Evidence Engine v5 — обновлённый клиент Weaviate (API 2.x):
    ✔ client.collections
    ✔ .data.insert / .data.update
    ✔ batch = client.batch.dynamic
    ✔ properties вместо "data"
    ✔ корректный schema ensure
    """

    def __init__(self, url: str):
        self.client = weaviate.Client(url)
        logger.info(f"🔗 VectorClient подключён к {url}")
        self.ensure_schema()
        self._configure_batch()

    # --------------------------------------------------------------------
    # Batch-config
    # --------------------------------------------------------------------
    def _configure_batch(self):
        try:
            self.client.batch.configure(
                batch_size=200,
                dynamic=True,
                timeout_retries=3,
            )
            logger.info("⚙ Batch вставка Weaviate настроена.")
        except Exception as e:
            logger.error(f"❌ Ошибка batch config: {e}")

    # --------------------------------------------------------------------
    # Schema ensure
    # --------------------------------------------------------------------
    def ensure_schema(self):
        try:
            schema = self.client.schema.get()
            classes = [c["class"] for c in schema.get("classes", [])]

            if "Chunk" in classes:
                return

            logger.warning("⚠ Schema Chunk отсутствует → создаём.")

            schema = {
                "class": "Chunk",
                "description": "Legal evidence chunk",
                "properties": [
                    {"name": "file_id", "dataType": ["string"]},
                    {"name": "page", "dataType": ["int"]},
                    {"name": "chunk_id", "dataType": ["string"]},
                    {"name": "text", "dataType": ["text"]},
                ],
                "vectorizer": "text2vec-transformers",
            }

            self.client.schema.create_class(schema)
            logger.info("✔ Schema Chunk создана.")

        except Exception as e:
            logger.error(f"❌ Ошибка создания schema Chunk: {e}")

    # --------------------------------------------------------------------
    # Batch INSERT (новый формат)
    # --------------------------------------------------------------------
    def batch_insert_chunk(
        self,
        text: str,
        file_id: str,
        page: int,
        chunk_id: str,
    ) -> bool:

        try:
            self.client.batch.add_data_object(
                class_name="Chunk",
                properties={
                    "file_id": file_id,
                    "page": page,
                    "chunk_id": chunk_id,
                    "text": text,
                }
            )
            return True

        except Exception as e:
            logger.error(f"[batch_insert_chunk] Ошибка: {e}")
            return False

    # --------------------------------------------------------------------
    # Flush
    # --------------------------------------------------------------------
    def flush(self) -> bool:
        try:
            self.client.batch.flush()
            return True
        except Exception as e:
            logger.error(f"[flush] Ошибка: {e}")
            return False

    # --------------------------------------------------------------------
    # Single INSERT (новый формат)
    # --------------------------------------------------------------------
    def insert_chunk(self, text: str, file_id: str, page: int, chunk_id: str) -> bool:
        """
        Fallback — одиночная вставка, используется если batch не сработал.
        """

        try:
            self.client.data_object.create(
                class_name="Chunk",
                properties={
                    "file_id": file_id,
                    "page": page,
                    "chunk_id": chunk_id,
                    "text": text,
                }
            )
            return True

        except Exception as e:
            logger.error(f"[insert_chunk] Ошибка: {e}")
            return False

    # --------------------------------------------------------------------
    # Search
    # --------------------------------------------------------------------
    def search(
        self,
        query_text: str,
        limit: int = 10,
        with_vector: bool = False,
    ) -> Dict[str, Any]:

        try:
            q = (
                self.client.query
                .get("Chunk", ["file_id", "page", "chunk_id", "text"])
                .with_near_text({"concepts": [query_text]})
                .with_limit(limit)
            )

            if with_vector:
                q = q.with_additional(["vector", "distance"])

            return q.do()

        except Exception as e:
            logger.error(f"[search] Ошибка Weaviate: {e}")
            return {}
