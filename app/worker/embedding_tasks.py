# app/worker/embedding_tasks.py

import logging
from typing import List
from uuid import UUID

from celery import shared_task

from app.worker.celery_app import celery_app
from app.db.session import SessionLocal
from app.db.models import Chunk
from app.search.vector_client import get_vector_client

logger = logging.getLogger("CELERY-EMBEDDINGS")


# ============================================================
# Индексация одного чанка
# ============================================================

@celery_app.task(name="app.worker.embedding_tasks.index_chunk")
def index_chunk(chunk_id: str):
    """
    Асинхронная индексация одного чанка в Weaviate.
    """
    # --- Проверяем UUID ---
    try:
        chunk_uuid = UUID(chunk_id)
    except Exception:
        logger.error(f"[index_chunk] Некорректный chunk_id: {chunk_id}")
        return

    db = SessionLocal()

    try:
        chunk: Chunk = (
            db.query(Chunk)
            .filter(Chunk.chunk_id == chunk_uuid)
            .first()
        )

        if not chunk:
            logger.warning(f"[index_chunk] Чанк {chunk_id} не найден")
            return

        vc = get_vector_client()
        vc.insert_chunk(
            text=chunk.text,
            file_id=str(chunk.file_id),
            page=chunk.page,
            chunk_id=str(chunk.chunk_id),
        )

        logger.info(f"[index_chunk] ✔ Индексирован чанк {chunk_id}")

    except Exception as e:
        logger.error(
            f"[index_chunk] ❌ Ошибка индексации чанка {chunk_id}: {e}",
            exc_info=True
        )

    finally:
        db.close()


# ============================================================
# Индексация всех чанков файла
# ============================================================

@celery_app.task(name="app.worker.embedding_tasks.index_file_chunks")
def index_file_chunks(file_id: str):
    """
    Индексация ВСЕХ чанков, принадлежащих одному файлу.
    Удобно для reindex после очистки Weaviate.
    """

    # --- Проверяем UUID ---
    try:
        file_uuid = UUID(file_id)
    except Exception:
        logger.error(f"[index_file_chunks] Некорректный file_id: {file_id}")
        return

    db = SessionLocal()

    try:
        chunks: List[Chunk] = (
            db.query(Chunk)
            .filter(Chunk.file_id == file_uuid)
            .order_by(Chunk.page.asc())
            .all()
        )

        if not chunks:
            logger.warning(f"[index_file_chunks] Нет чанков для файла {file_id}")
            return

        vc = get_vector_client()

        for ch in chunks:
            vc.insert_chunk(
                text=ch.text,
                file_id=str(ch.file_id),
                page=ch.page,
                chunk_id=str(ch.chunk_id),
            )

        logger.info(
            f"[index_file_chunks] ✔ Индексировано {len(chunks)} чанков файла {file_id}"
        )

    except Exception as e:
        logger.error(
            f"[index_file_chunks] ❌ Ошибка индексации файла {file_id}: {e}",
            exc_info=True
        )

    finally:
        db.close()
