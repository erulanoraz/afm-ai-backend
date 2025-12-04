# app/tasks/vector_tasks.py
import logging
import time
from app.worker.celery_app import celery_app
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.db.models import Chunk
from app.search.vector_client import get_vector_client

logger = logging.getLogger("CELERY-VECTORS")


@celery_app.task(
    name="app.tasks.vector_tasks.enqueue_chunk_vectorization",
    bind=True,
    max_retries=5
)
def enqueue_chunk_vectorization(self, chunk_id: str):
    """
    Vector Pipeline 6.0 ИСПРАВЛЕННАЯ:
    • ждём commit от ingest
    • безопасная вставка в Weaviate
    • ✅ ВЫЗЫВАЕМ FLUSH!
    """
    db: Session = SessionLocal()

    try:
        # ======== ЖДЁМ КОГДА INGEST СДЕЛАЕТ COMMIT ==========
        for attempt in range(5):
            chunk = db.query(Chunk).filter(Chunk.chunk_id == chunk_id).first()
            if chunk:
                break
            logger.warning(f"⏳ Chunk {chunk_id} not committed yet (attempt={attempt+1}/5)")
            time.sleep(0.5)
        else:
            logger.error(f"❌ Chunk NOT FOUND после 5 попыток: {chunk_id}")
            return

        # ========== Отправляем в Weaviate ==========
        vc = get_vector_client()
        
        # Используем batch API
        ok = vc.insert_chunk(
            text=chunk.text,
            file_id=str(chunk.file_id),
            page=chunk.page or 0,
            chunk_id=str(chunk_id),
        )


        if not ok:
            raise Exception("Weaviate batch insertion failed")

        # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: ВЫЗЫВАЕМ FLUSH!
        flush_ok = vc.flush()
        if not flush_ok:
            logger.warning(f"⚠️ Flush не удался для chunk {chunk_id}, но продолжаем")

        logger.info(f"✔ Vectorized chunk={chunk_id}")

    except Exception as e:
        logger.error(f"❌ Vector indexing error for {chunk_id}: {e}")
        raise self.retry(exc=e, countdown=2)

    finally:
        db.close()

