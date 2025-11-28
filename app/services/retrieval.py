# app/services/retrieval.py
import logging
import re
from typing import List, Dict, Any, Optional
from uuid import UUID
from sqlalchemy.orm import Session

from app.db.models import File, Chunk

logger = logging.getLogger(__name__)

# ============================================================
# 🔥 ИСПРАВЛЕННАЯ Retrieval 6.0 — ФОКУС НА КАЧЕСТВО
# ============================================================

# Не берём ВСЕ — берём только высокорелевантные
TOP_K_WIDE = 200        # ← было 600! СЛИШКОМ МНОГО
TOP_BASELINE_LIMIT = 150
TOP_RERANK_INPUT = 80   # ← было 300, отправляем мало но качественно


# ============================================================
# 🔥 НОРМАЛИЗАЦИЯ (убираем мусор)
# ============================================================

def normalize_text(text: str) -> str:
    """Убираем OCR мусор и техничку"""
    if not text:
        return ""

    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    # Удаляем техмусор
    garbage = [
        r"©\s?Все права защищены",
        r"сканировано\s?с\s?помощью.*",
        r"страница\s?\d+\s?из\s?\d+",
        r"Документ создан.*",
        r"QR[- ]?код.*",
        r"электронный документ.*",
        r"Просмотрено.*",
        r"Дата печати.*",
        r"хеш.*",
        r"эцп.*",
    ]

    for g in garbage:
        text = re.sub(g, "", text, flags=re.IGNORECASE)

    return text.strip()


# ============================================================
# 🔥 CRISP baseline weight (ГЛАВНОЕ УЛУЧШЕНИЕ)
# ============================================================

def baseline_weight(filename: str, text: str) -> float:
    """
    Строгий baseline — НЕ берём мусор.
    Только файлы с РЕАЛЬНЫМ содержанием.
    """
    fn = filename.lower()
    t = text.lower()

    # VERY STRONG — допросы подозреваемого (THE GOLD)
    if any(x in fn for x in ["протокол_допроса_подозреваем", "допроса подозреваем", "куи"]):
        return 0.99

    # STRONG — рапорты, ердр
    if any(x in fn for x in ["рапорт", "ердр", "постановление о возбуждении"]):
        return 0.90

    # MEDIUM — допросы потерпевших, свидетелей
    if any(x in fn for x in ["допроса_потерпевш", "допроса потерпевш", "свидетелей"]):
        return 0.75

    # WEAK — просто постановления
    if "постановление" in fn:
        return 0.60

    # GARBAGE — файлы содержат только техмусор
    if len(t) < 50 or t.count(" ") < 5:
        return 0.0

    # DEFAULT — остальное с малым весом
    return 0.35


# ============================================================
# 🔥 ГЛАВНЫЙ RETRIEVAL (исправленный)
# ============================================================

def get_file_docs_for_qualifier(
    db: Session,
    file_ids: Optional[List[str]] = None,
    case_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieval 6.0 — жёсткая фильтрация, НО качественная выборка.
    """

    query = db.query(File)

    if case_id:
        query = query.filter(File.case_id == case_id)

    if file_ids:
        query = query.filter(File.file_id.in_(file_ids))

    files = query.all()
    logger.info(f"📄 Retrieval 6.0: всего файлов = {len(files)}")

    docs: List[Dict[str, Any]] = []

    # ============================================================
    # Читаем ТОЛЬКО релевантные чанки
    # ============================================================
    for f in files:
        file_id = str(f.file_id)
        filename = (f.filename or "").lower()

        # 🔴 ФИЛЬТР 1: пропускаем файлы которые явно мусор
        weight = baseline_weight(filename, "")
        if weight < 0.30:
            logger.debug(f"⏭️ Пропуск: {filename} (weight={weight})")
            continue

        try:
            chunks = (
                db.query(Chunk)
                .filter(Chunk.file_id == UUID(file_id))
                .order_by(
                    Chunk.page.asc(),
                    Chunk.start_offset.asc(),
                )
                .all()
            )
        except Exception as e:
            logger.error(f"❌ Ошибка чанков {file_id}: {e}")
            continue

        if not chunks:
            continue

        # 🔴 ФИЛЬТР 2: пропускаем пустые и мусорные чанки
        for ch in chunks:
            raw = (ch.text or "").strip()
            
            # Слишком коротко?
            if len(raw) < 30:
                continue
            
            # Только служебная инфо?
            if raw.count(" ") < 3:
                continue

            clean = normalize_text(raw)
            
            if not clean or len(clean) < 20:
                continue

            # ✅ Добавляем только хорошие чанки
            docs.append({
                "file_id": file_id,
                "filename": f.filename,
                "page": ch.page or 1,
                "chunk_id": str(ch.chunk_id),
                "text": clean,
            })

    logger.info(f"📦 После фильтра: {len(docs)} чанков")

    if not docs:
        return []

    # ============================================================
    # Baseline сортировка
    # ============================================================
    for d in docs:
        d["baseline_weight"] = baseline_weight(
            filename=d["filename"],
            text=d["text"],
        )

    # 🔴 ЖЁСТКАЯ СОРТИРОВКА
    docs = sorted(docs, key=lambda x: x["baseline_weight"], reverse=True)
    docs = docs[:TOP_BASELINE_LIMIT]

    logger.info(f"✅ Retrieval 6.0: передаём {min(len(docs), TOP_RERANK_INPUT)} документов в reranker")

    return docs[:TOP_RERANK_INPUT]


# ============================================================
# 🔍 DEBUG SEARCH (без изменений)
# ============================================================

def search_chunks(db: Session, query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Упрощённый поиск для debug API."""

    if not query or not query.strip():
        return []

    pattern = f"%{query.strip()}%"

    try:
        chunks = (
            db.query(Chunk)
            .filter(Chunk.text.ilike(pattern))
            .order_by(Chunk.page.asc())
            .limit(limit)
            .all()
        )
    except Exception as e:
        logger.error(f"[search_chunks ERROR] {e}")
        return []

    results = []
    for ch in chunks:
        results.append({
            "file_id": str(ch.file_id),
            "chunk_id": str(ch.chunk_id),
            "page": ch.page,
            "text": ch.text[:300] + ("..." if len(ch.text) > 300 else "")
        })

    return results