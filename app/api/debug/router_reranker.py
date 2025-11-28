# app/api/debug/router_reranker.py
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from app.services.reranker import LLMReranker

router = APIRouter(prefix="/debug/reranker", tags=["DEBUG – Reranker"])


class RerankItem(BaseModel):
    text: str = Field(..., description="Текст чанка")
    meta: Dict[str, Any] = Field(default_factory=dict)


class RerankRequest(BaseModel):
    query: str
    items: List[RerankItem]


@router.post("/", summary="Тестирование Reranker 3.0")
async def debug_reranker(req: RerankRequest):
    """
    Debug Reranker:
    - принимает query
    - принимает список items
    - прогоняет через LLMReranker
    - показывает итоговый порядок + score для каждого чанка
    """

    reranker = LLMReranker()

    # Конвертируем внутрь стандартного формата
    items_internal = [
        {
            "text": it.text,
            "meta": it.meta,
            # позволяем через meta при желании передать filename / page / evidence
            "filename": it.meta.get("filename"),
            "page": it.meta.get("page"),
            "evidence": it.meta.get("evidence"),
        }
        for it in req.items
    ]

    ranked = reranker.rerank(
        query=req.query,
        items=items_internal
    )

    formatted = []
    for idx, it in enumerate(ranked, start=1):
        baseline = float(it.get("baseline_score", 0.0))
        llm = float(it.get("llm_score", 0.0))
        cross = float(it.get("cross_score", baseline + llm))

        formatted.append({
            "rank": idx,
            "score": round(cross, 4),              # общий итоговый скор
            "baseline_score": round(baseline, 4),  # что дал baseline
            "llm_score": round(llm, 4),            # что добавил LLM
            "text_preview": it.get("text", "")[:250],
            "full_text_length": len(it.get("text", "")),
            "file_id": it.get("file_id"),
            "page": it.get("page"),
            "chunk_id": it.get("chunk_id"),
            "meta": it.get("meta", {}),
        })

    return {
        "query": req.query,
        "items_in": len(req.items),
        "items_out": len(ranked),
        "ranked": formatted,
    }
