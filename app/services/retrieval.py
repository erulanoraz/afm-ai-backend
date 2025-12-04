import logging
import re
from typing import List, Dict, Any, Optional
from uuid import UUID
from sqlalchemy.orm import Session

from app.db.models import File, Chunk

logger = logging.getLogger(__name__)


# ============================================================
# 🔥 RETRIEVAL 7.6 — GLOBAL COVERAGE (NO CASE_ID FILTER)
# ============================================================
#
# • Полное отключение фильтрации по case_id
# • Файлы без case_id НЕ пропускаются
# • Привязка только по file_ids (если переданы)
# • Если file_ids пусто → используются ВСЕ файлы в БД
# ============================================================

TOP_BASELINE_LIMIT = 400
TOP_RERANK_INPUT = 300
MIN_TEXT_LENGTH = 20


# ============================================================
# 🔧 Текстовая нормализация
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
        r"хеш.*",
        r"эцп.*",
    ]

    for g in garbage:
        text = re.sub(g, "", text, flags=re.IGNORECASE)

    return text.strip()


# ============================================================
# 🔧 Мягкий baseline (ранжирование, НЕ фильтр!)
# ============================================================

def baseline_weight(filename: str, text: str) -> float:
    fn = (filename or "").lower()
    t = (text or "").lower()

    # базовый вес по имени файла
    if any(x in fn for x in [
        "допрос", "протокол_допроса", "допроса",
        "объяснение", "пояснени"
    ]):
        weight = 0.95

    elif any(x in fn for x in ["рапорт", "ердр"]):
        weight = 0.90

    elif "постановление" in fn:
        weight = 0.80

    elif "приложение" in fn:
        weight = 0.60

    else:
        weight = 0.50

    # усиливаем документы, где фигурирует подозреваемый / обвиняемый
    if any(x in t for x in ["подозреваем", "обвиняем", "сообщение о подозрении"]):
        weight = max(weight, 0.97)

    # слегка опускаем «заявление потерпевшего»
    if "заявление" in t and "потерпевш" in t:
        weight = min(weight, 0.75)

    return weight


# ============================================================
# 🔧 Фильтр вопросительных блоков
# ============================================================

def _is_question_block(text: str) -> bool:
    if not text:
        return False

    low = text.lower().strip()

    if "вопрос:" in low or "вопрос :" in low or "вопрос " in low:
        return True

    if "спросил" in low or "спросила" in low or "каким образом" in low:
        return True

    if low.endswith("?"):
        return True

    # многострочные блоки с вопросами
    if "?" in text and "\n" in text:
        return True

    return False


# ============================================================
# 🔥 ГЛАВНАЯ ФУНКЦИЯ RETRIEVAL 7.6 (NO CASE FILTER)
# ============================================================

def get_file_docs_for_qualifier(
    db: Session,
    file_ids: Optional[List[str]] = None,
    case_id: Optional[str] = None,  # ← игнорируется
) -> List[Dict[str, Any]]:

    query = db.query(File)

    # ❌ БОЛЬШЕ НЕТ:
    # if case_id:
    #    query = query.filter(File.case_id == case_id)

    # ✔ Если переданы file_ids → используем их
    if file_ids:
        query = query.filter(File.file_id.in_(file_ids))
        logger.info(f"📄 Retrieval 7.6: используем file_ids ({len(file_ids)})")
    else:
        logger.info(f"📄 Retrieval 7.6: file_ids не переданы → используем ВСЕ файлы.")

    files = query.all()
    logger.info(f"📄 Retrieval 7.6: всего файлов = {len(files)}")

    docs: List[Dict[str, Any]] = []

    # ============================================================
    # Читаем все файлы (включая файлы без case_id)
    # ============================================================

    for f in files:
        file_id = str(f.file_id)
        filename = (f.filename or "").lower()

        try:
            chunks = (
                db.query(Chunk)
                .filter(Chunk.file_id == UUID(file_id))
                .order_by(Chunk.page.asc(), Chunk.start_offset.asc())
                .all()
            )
        except Exception as e:
            logger.error(f"❌ Ошибка чтения чанков для {file_id}: {e}")
            continue

        if not chunks:
            continue

        for ch in chunks:
            raw = (ch.text or "").strip()

            if len(raw) < MIN_TEXT_LENGTH:
                continue

            if raw.count(" ") < 3:
                continue

            clean = normalize_text(raw)
            if not clean or len(clean) < MIN_TEXT_LENGTH:
                continue

            if _is_question_block(clean):
                continue

            docs.append({
                "file_id": file_id,
                "filename": f.filename,
                "page": ch.page or 1,
                "chunk_id": str(ch.chunk_id),
                "text": clean,
            })

    logger.info(f"📦 Retrieval 7.6: количество чанков после фильтров = {len(docs)}")

    if not docs:
        return []

    # ============================================================
    # baseline сортировка
    # ============================================================

    for d in docs:
        d["baseline_weight"] = baseline_weight(d["filename"], d["text"])

    docs = sorted(docs, key=lambda x: x["baseline_weight"], reverse=True)

    docs = docs[:TOP_BASELINE_LIMIT]

    logger.info(
        f"✅ Retrieval 7.6: передаём {min(len(docs), TOP_RERANK_INPUT)} документов в RAG Router"
    )

    return docs[:TOP_RERANK_INPUT]


# ============================================================
# 🔍 DEBUG SEARCH
# ============================================================

def search_chunks(db: Session, query: str, limit: int = 20) -> List[Dict[str, Any]]:
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
