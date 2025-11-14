# app/services/retrieval.py

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from uuid import UUID

from app.db.models import File, Chunk

logger = logging.getLogger(__name__)


# ============================================================
# 🔥 Основная функция retrieval — EXTRACTOR-READY FORMAT
# ============================================================

def get_file_docs_for_qualifier(
    db: Session,
    file_ids: Optional[List[str]] = None,
    case_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Возвращает документы строго в формате:

        {
            "file_id": "uuid",
            "page": 1,
            "chunk_id": "uuid",
            "text": "..."
        }

    Этот формат является обязательным для:
    - roles extractor
    - events extractor
    - timeline builder
    - legal facts extractor
    - inline citations
    """

    query = db.query(File)

    if case_id:
        query = query.filter(File.case_id == case_id)

    if file_ids:
        query = query.filter(File.file_id.in_(file_ids))

    files = query.all()
    logger.info(f"📄 Retrieval: найдено файлов: {len(files)}")

    docs: List[Dict[str, Any]] = []

    for f in files:
        file_id_str = str(f.file_id)

        # Чанки с правильной сортировкой
        try:
            chunks = (
                db.query(Chunk)
                .filter(Chunk.file_id == UUID(file_id_str))     # правильное сравнение UUID
                .order_by(
                    Chunk.page.asc(),
                    Chunk.start_offset.asc()
                )
                .all()
            )
        except Exception as e:
            logger.error(f"❌ Ошибка получения чанков для файла {file_id_str}: {e}")
            chunks = []

        if not chunks:
            logger.warning(f"⚠️ Файл {file_id_str} не содержит чанков — пропускаю.")
            continue

        # Преобразование чанков в EXTRACTOR-ready формат
        for ch in chunks:
            text = getattr(ch, "text", None) or getattr(ch, "content", None) or ""

            docs.append({
                "file_id": file_id_str,
                "page": ch.page or 1,
                "chunk_id": str(ch.chunk_id),
                "text": text.strip(),
            })

    logger.info(f"📦 Retrieval вернул {len(docs)} документов")

    return docs


# ============================================================
# 🔹 Вспомогательная функция: получение всех чанков файла
# ============================================================

def get_chunks_by_file_id(db: Session, file_id: str) -> List[Dict[str, Any]]:
    """Безопасно возвращает список чанков с нормализацией данных."""

    try:
        chunks = (
            db.query(Chunk)
            .filter(Chunk.file_id == UUID(file_id))
            .order_by(
                Chunk.page.asc(),
                Chunk.start_offset.asc()
            )
            .all()
        )
    except Exception as e:
        logger.error(f"Ошибка get_chunks_by_file_id для {file_id}: {e}")
        return []

    result = []

    for ch in chunks:
        text = getattr(ch, "text", None) or getattr(ch, "content", None) or ""

        result.append({
            "chunk_id": str(ch.chunk_id),
            "file_id": file_id,
            "page": ch.page or 1,
            "text": text,
            "metadata": {
                "start_offset": getattr(ch, "start_offset", None),
                "created_at": getattr(ch, "created_at", None),
            },
        })

    if not result:
        logger.warning(f"⚠️ Файл {file_id} вернул 0 чанков — создаю placeholder.")
        return [{
            "chunk_id": f"{file_id}-empty",
            "file_id": file_id,
            "page": 1,
            "text": "",
            "metadata": {}
        }]

    return result


# ============================================================
# 🔹 Статистика по делу
# ============================================================

def get_file_text_stats(db: Session, case_id: str) -> Dict[str, Any]:
    """Возвращает статистику по файлам и чанкам в деле."""

    try:
        files = db.query(File).filter(File.case_id == case_id).all()

        stats = {
            "case_id": case_id,
            "total_files": len(files),
            "files_with_chunks": 0,
            "total_chunks": 0,
            "total_chars": 0,
            "files": [],
        }

        for f in files:
            file_id_str = str(f.file_id)

            chunks = db.query(Chunk).filter(
                Chunk.file_id == UUID(file_id_str)
            ).all()

            text_length = sum(
                len(getattr(c, "text", "") or "")
                for c in chunks
            )

            stats["total_chunks"] += len(chunks)
            stats["total_chars"] += text_length

            if chunks:
                stats["files_with_chunks"] += 1

            stats["files"].append({
                "file_id": file_id_str,
                "filename": getattr(f, "filename", None),
                "chunks": len(chunks),
                "text_length": text_length,
            })

        logger.info(
            f"📊 Статистика: {stats['total_files']} файлов, "
            f"{stats['total_chunks']} chunks, "
            f"{stats['total_chars']} символов"
        )

        return stats

    except Exception as e:
        logger.error(f"Ошибка get_file_text_stats: {e}")
        return {"error": str(e)}
