# app/services/retrieval.py
import logging
import re
from typing import List, Dict, Any, Optional
from uuid import UUID
from sqlalchemy.orm import Session

from app.db.models import File, Chunk

logger = logging.getLogger(__name__)


# ============================================================
# 🔥 Нормализация текста (Kazakhstan legal safe)
# ============================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    garbage = [
        r"©\s?Все права защищены",
        r"сканировано\s?с\s?помощью.*",
        r"страница\s?\d+\s?из\s?\d+",
        r"Документ создан.*",
        r"QR[- ]?код.*",
        r"электронный документ.*",
        r"Просмотрено.*",
        r"Дата печати.*",
    ]

    for g in garbage:
        text = re.sub(g, "", text, flags=re.IGNORECASE)

    return text.strip()


# ============================================================
# 🔥 Оценка чанка (baseline weight)
# ============================================================

def baseline_weight(filename: str, text: str) -> float:
    fn = filename.lower()
    t = text.lower()

    # супер важные документы
    strong = [
        "рапорт", "куи", "ердр", "протокол_допроса_подозреваем",
        "протокол допроса подозреваем",
    ]
    medium = [
        "протокол_допроса_потерпевшего",
        "протокол_допроса_потерпевш",
        "постановление о признании лица потерпевшим",
        "постановление о признании лица гражданским истцом",
    ]

    if any(x in fn for x in strong):
        return 1.0

    if any(x in t for x in ["он подозревается", "она подозревается"]):
        return 0.95

    if any(x in fn for x in medium):
        return 0.80

    if "постановление" in fn:
        return 0.70

    return 0.40


# ============================================================
# 🔥 Retrieval 4.0 — главный
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
            continue

        for ch in chunks:
            raw_text = (ch.text or "").strip()
            if not raw_text:
                continue

            clean = normalize_text(raw_text)
            if not clean:
                continue

            docs.append({
                "file_id": file_id,
                "filename": f.filename,
                "page": ch.page or 1,
                "chunk_id": str(ch.chunk_id),
                "text": clean,
            })

    # ===========================================================
    # 🔥 BASELINE сортировка по важности документа
    # ===========================================================
    for d in docs:
        d["baseline_weight"] = baseline_weight(
            filename=d["filename"],
            text=d["text"]
        )

    docs = sorted(docs, key=lambda x: x["baseline_weight"], reverse=True)

    # Ограничение — не более 400 чанков
    docs = docs[:400]

    logger.info(f"📦 Retrieval 4.0 вернул документов: {len(docs)}")
    return docs
