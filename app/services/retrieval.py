# app/services/retrieval.py
import logging
import re
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from uuid import UUID

from app.db.models import File, Chunk

logger = logging.getLogger(__name__)


# ============================================================
# 🧼 Нормализация текста (Kazakhstan-ready, безопасная)
# ============================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""

    # normalize newlines, spaces
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    # ⚠️ Удаляем ТОЛЬКО технический шум, НЕ фабулу
    garbage = [
        r"©\s?Все права защищены",
        r"сканировано\s?с\s?помощью.*",
        r"страница\s?\d+\s?из\s?\d+",
        r"Документ создан.*",
        r"QR[- ]?код.*",
        r"электронный документ.*",
        r"Просмотрено на.*",
        r"Дата печати.*",
        # подпись разрешено оставлять — важно
    ]

    for g in garbage:
        text = re.sub(g, "", text, flags=re.IGNORECASE)

    return text.strip()



# ============================================================
# 🧠 Лемматизация RU/KZ — безопасная
# ============================================================

def lemmatize(text: str) -> str:
    # пока просто нормализуем
    return normalize_text(text)



# ============================================================
# 🔥 Главная функция Retrieval 3.1
# ============================================================

def get_file_docs_for_qualifier(
    db: Session,
    file_ids: Optional[List[str]] = None,
    case_id: Optional[str] = None,
) -> List[Dict[str, Any]]:

    query = db.query(File)

    if case_id:
        query = query.filter(File.case_id == case_id)

    if file_ids:
        query = query.filter(File.file_id.in_(file_ids))

    files = query.all()
    logger.info(f"📄 Retrieval: найдено файлов = {len(files)}")

    docs: List[Dict[str, Any]] = []

    for f in files:
        file_id = str(f.file_id)

        try:
            chunks = (
                db.query(Chunk)
                .filter(Chunk.file_id == UUID(file_id))
                .order_by(
                    Chunk.page.asc(),
                    Chunk.start_offset.asc(),
                    Chunk.created_at.asc(),
                    Chunk.chunk_id.asc(),
                )
                .all()
            )
        except Exception as e:
            logger.error(f"❌ Ошибка получения чанков файла {file_id}: {e}")
            continue

        if not chunks:
            logger.warning(f"⚠️ Файл {file_id} пуст — пропускаю.")
            continue

        for ch in chunks:
            raw_text = getattr(ch, "text", "") or ""
            clean_text = lemmatize(raw_text)

            if not clean_text.strip():
                continue

            docs.append({
                "file_id": file_id,
                "page": ch.page or 1,
                "chunk_id": str(ch.chunk_id),
                "text": clean_text,
            })

    # -----------------------------
    # 🍀 Лог после наполнения docs
    # -----------------------------
    logger.info("=== RETRIEVAL OUTPUT START ===")
    for d in docs[:20]:
        txt = d.get("text", "").replace("\n", " ")
        logger.info(f"PAGE={d.get('page')} | LEN={len(txt)} | {txt[:300]}")
    logger.info("=== RETRIEVAL OUTPUT END ===")

    logger.info(f"📦 Retrieval 3.1 вернул документов: {len(docs)}")
    return docs



# ============================================================
# 🔹 Чанки по file_id
# ============================================================

def get_chunks_by_file_id(db: Session, file_id: str) -> List[Dict[str, Any]]:
    try:
        chunks = (
            db.query(Chunk)
            .filter(Chunk.file_id == UUID(file_id))
            .order_by(
                Chunk.page.asc(),
                Chunk.start_offset.asc(),
                Chunk.created_at.asc(),
            )
            .all()
        )
    except Exception as e:
        logger.error(f"❌ Ошибка get_chunks_by_file_id({file_id}): {e}")
        return []

    result = []

    for ch in chunks:
        clean_text = lemmatize(getattr(ch, "text", "") or "")
        result.append({
            "chunk_id": str(ch.chunk_id),
            "file_id": file_id,
            "page": ch.page or 1,
            "text": clean_text,
            "metadata": {
                "start_offset": getattr(ch, "start_offset", None),
                "created_at": getattr(ch, "created_at", None),
            },
        })

    if not result:
        logger.warning(f"⚠️ Файл {file_id} вернул 0 чанков. Создаю placeholder")
        return [{
            "chunk_id": f"{file_id}-empty",
            "file_id": file_id,
            "page": 1,
            "text": "",
            "metadata": {}
        }]

    return result



# ============================================================
# 📊 Статистика (улучшенная)
# ============================================================

def get_file_text_stats(db: Session, case_id: str) -> Dict[str, Any]:
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
            file_id = str(f.file_id)

            chunks = (
                db.query(Chunk)
                (Chunk.file_id == UUID(file_id))
                .all()
            )

            total_text = sum(len(c.text or "") for c in chunks)

            stats["total_chunks"] += len(chunks)
            stats["total_chars"] += total_text
            stats["files_with_chunks"] += 1 if chunks else 0

            stats["files"].append({
                "file_id": file_id,
                "filename": f.filename,
                "chunks": len(chunks),
                "text_length": total_text,
            })

        logger.info(
            f"📊 Retrieval Stats: файлов={stats['total_files']}, "
            f"чанков={stats['total_chunks']}, "
            f"символов={stats['total_chars']}"
        )

        return stats

    except Exception as e:
        logger.error(f"❌ Ошибка get_file_text_stats: {e}")
        return {"error": str(e)}
