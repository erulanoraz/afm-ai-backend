import logging
from app.worker.celery_app import celery_app
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.db.models import Chunk
from app.search.vector_client import get_vector_client

logger = logging.getLogger("CELERY-VECTORS")

@celery_app.task(name="app.tasks.vector_tasks.enqueue_chunk_vectorization")
def enqueue_chunk_vectorization(chunk_id: str):

    """
    Celery: берём chunk из PostgreSQL → отправляем в Weaviate.
    """
    db: Session = SessionLocal()

    try:
        chunk = db.query(Chunk).filter(Chunk.chunk_id == chunk_id).first()

        if not chunk:
            logger.error(f"❌ Chunk not found: {chunk_id}")
            return

        vc = get_vector_client()
        vc.insert_chunk(
            text=chunk.text,
            file_id=str(chunk.file_id),
            page=chunk.page,
            chunk_id=chunk_id,
        )

        logger.info(f"✔ Indexed in Weaviate: chunk={chunk_id}")

    except Exception as e:
        logger.error(f"❌ Vector indexing error for {chunk_id}: {e}")

    finally:
        db.close()
