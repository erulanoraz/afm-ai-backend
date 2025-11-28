# app/api/debug/router_retrieval.py
import logging
from typing import List, Dict, Any

import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.services.retrieval import get_file_docs_for_qualifier
from app.services.embeddings import embed_text

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/debug/retrieval",
    tags=["DEBUG – Retrieval"]
)


# ============================
# 📥 Модель запроса
# ============================

class RetrievalRequest(BaseModel):
    case_id: str
    query: str
    top_k: int = 20


# ============================
# 🔢 Cosine similarity
# ============================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(a.dot(b) / denom)


# ============================
# 🔢 Нормализация baseline weight
# ============================

def normalize_baseline_weight(w: float) -> float:
    """
    Приводим baseline_weight к диапазону [0, 1],
    чтобы можно было смешивать с cosine score.
    """
    if w <= 0:
        return 0.0
    # 1.5 — условный "максимальный" вес, можно подстроить под практику
    return float(max(0.0, min(1.0, w / 1.5)))


# ============================
# 🧠 DEBUG Retrieval 5.1
# ============================

@router.post(
    "/",
    summary="Проверка Retrieval 5.1 (baseline + семантический rerank по чанкам)",
)
async def debug_retrieval(
    req: RetrievalRequest,
    db: Session = Depends(get_db),
):
    """
    Полный debug Retrieval-пайплайна:

    1) Забираем чанки Retrieval 5.0 (baseline отбор по делу)
    2) Строим embedding для query
    3) Для каждого чанка строим embedding текста (усечённо)
    4) Считаем cosine similarity(query, chunk)
    5) Комбинируем baseline_weight и cosine score → final_score
    6) Сортируем по final_score и отдаём top-K чанков

    Это максимально приближено к боевому режиму:
    - используется тот же Retrieval 5.0 (get_file_docs_for_qualifier)
    - используется тот же embed_text(), что и в основном RAG
    - ранжирование идёт по комбинированному скору baseline + семантика
    """

    # ---------------------------------------
    # 1) Retrieval 5.0 — baseline документы
    # ---------------------------------------
    docs: List[Dict[str, Any]] = get_file_docs_for_qualifier(
        db,
        case_id=req.case_id,
    )

    if not docs:
        return {
            "case_id": req.case_id,
            "query": req.query,
            "error": "no_docs",
            "message": "Retrieval 5.0 не нашёл документов по делу",
        }

    baseline_count = len(docs)
    logger.info(f"[DEBUG Retrieval] Baseline docs: {baseline_count}")

    # ---------------------------------------
    # 2) Query embedding
    # ---------------------------------------
    try:
        q_vec_list = embed_text(req.query)
    except Exception as e:
        logger.error(f"[DEBUG Retrieval] Ошибка embed_text для query: {e}")
        return {
            "case_id": req.case_id,
            "query": req.query,
            "error": "embedding_error",
            "message": f"Ошибка при генерации embedding для запроса: {e}",
        }

    if not q_vec_list:
        return {
            "case_id": req.case_id,
            "query": req.query,
            "error": "embedding_error",
            "message": "Embedding-сервис вернул пустой вектор для запроса",
        }

    q_vec = np.array(q_vec_list, dtype=np.float32)

    # ---------------------------------------
    # 3) Для каждого чанка считаем cosine
    # ---------------------------------------
    results: List[Dict[str, Any]] = []

    for d in docs:
        text = d["text"] or ""
        if not text.strip():
            continue

        # ограничиваем текст, чтобы не перегружать embedding-модель
        chunk_text = text[:800]

        try:
            chunk_vec_list = embed_text(chunk_text)
        except Exception as e:
            logger.error(
                f"[DEBUG Retrieval] Ошибка embed_text для чанка "
                f"{d.get('file_id')}:{d.get('chunk_id')}: {e}"
            )
            continue

        if not chunk_vec_list:
            # пропускаем чанки без вектора
            continue

        chunk_vec = np.array(chunk_vec_list, dtype=np.float32)
        cosine_score = cosine_similarity(q_vec, chunk_vec)

        # baseline_weight уже рассчитан в get_file_docs_for_qualifier
        baseline_w = float(d.get("baseline_weight", 0.0))
        baseline_norm = normalize_baseline_weight(baseline_w)

        # комбинированный скор:
        #  - 60% семантика
        #  - 40% baseline (тип документа, evidence)
        final_score = 0.6 * cosine_score + 0.4 * baseline_norm

        results.append(
            {
                "file_id": d["file_id"],
                "filename": d["filename"],
                "page": d["page"],
                "chunk_id": d["chunk_id"],
                "baseline_weight": baseline_w,
                "baseline_norm": baseline_norm,
                "cosine_score": cosine_score,
                "final_score": final_score,
                "text": (
                    text[:400] + "..."
                    if len(text) > 400
                    else text
                ),
            }
        )

    if not results:
        return {
            "case_id": req.case_id,
            "query": req.query,
            "baseline_docs": baseline_count,
            "error": "no_semantic_results",
            "message": "Не удалось сгенерировать семантические оценки для чанков "
                       "(embedding-провайдер вернул пустые вектора).",
        }

    # ---------------------------------------
    # 4) Сортировка и top-K
    # ---------------------------------------
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    top_k = max(1, req.top_k)
    results = results[:top_k]

    return {
        "case_id": req.case_id,
        "query": req.query,
        "top_k": top_k,
        "baseline_docs": baseline_count,
        "returned": len(results),
        "results": results,
    }
