# app/search/vector_client.py

import logging
from typing import Any, Dict

import weaviate

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
    Vector Client 6.1 — стабильная версия
    Работает с Weaviate 1.25.8, text2vec-transformers
    """

    def __init__(self, url: str):

        # КЛЮЧЕВОЙ ФИКС:
        # отключить OIDC/ADMINLIST чтобы клиент не пытался получить openid-config
        self.client = weaviate.Client(
            url=url,
            auth_client_secret=None,
            additional_headers={},
            timeout_config=(5, 20),
        )

        logger.info(f"🔗 Weaviate client подключен к {url}")

        self.ensure_schema()
        self._configure_batch()

    # ===================================================================================
    # SCHEMA
    # ===================================================================================

    def ensure_schema(self) -> None:
        try:
            schema = self.client.schema.get()
            classes = [c["class"] for c in schema.get("classes", [])]

            if "Chunk" in classes:
                logger.info("✔ Schema 'Chunk' уже есть")
                return

            logger.warning("⚠ Schema 'Chunk' отсутствует → создаём")

            chunk_class = {
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

            self.client.schema.create_class(chunk_class)
            logger.info("✔ Schema 'Chunk' создана")

        except Exception as e:
            logger.error(f"❌ Ошибка при создании схемы Chunk: {e}")

    # ===================================================================================
    # BATCH
    # ===================================================================================

    def _configure_batch(self) -> None:
        try:
            self.client.batch.configure(
                batch_size=100,
                dynamic=True,
                timeout_retries=3,
            )
            logger.info("⚙ Batch режим включён")
        except Exception as e:
            logger.error(f"❌ Batch config error: {e}")

    def batch_insert_chunk(self, text: str, file_id: str, page: int, chunk_id: str) -> bool:
        try:
            self.client.batch.add_data_object(
                data_object={
                    "file_id": file_id,
                    "page": page,
                    "chunk_id": chunk_id,
                    "text": text,
                },
                class_name="Chunk",
                uuid=str(chunk_id),
            )
            return True

        except Exception as e:
            logger.error(f"❌ batch_insert_chunk({chunk_id}) error: {e}")
            return False

    def flush(self) -> bool:
        try:
            self.client.batch.flush()
            return True
        except Exception as e:
            logger.error(f"❌ flush error: {e}")
            return False

    # ===================================================================================
    # SINGLE INSERT
    # ===================================================================================

    def insert_chunk(self, text: str, file_id: str, page: int, chunk_id: str) -> bool:
        try:
            self.client.data_object.create(
                data_object={
                    "file_id": file_id,
                    "page": page,
                    "chunk_id": chunk_id,
                    "text": text,
                },
                class_name="Chunk",
                uuid=str(chunk_id),
            )
            return True

        except Exception as e:
            logger.error(f"❌ insert_chunk({chunk_id}) error: {e}")
            return False

    # ===================================================================================
    # SEARCH
    # ===================================================================================

    def search(self, query_text: str, limit: int = 10, with_vector: bool = False) -> Dict[str, Any]:
        try:
            q = (
                self.client.query
                .get("Chunk", ["file_id", "page", "chunk_id", "text"])
                .with_near_text({"concepts": [query_text]})
                .with_limit(limit)
            )

            if with_vector:
                q = q.with_additional(["vector", "distance"])

            result = q.do()
            return result

        except Exception as e:
            logger.error(f"❌ search error: {e}")
            return {}
