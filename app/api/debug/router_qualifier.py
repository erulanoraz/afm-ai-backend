from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Any, Dict

from sqlalchemy.orm import Session

from app.db import get_db
from app.services.retrieval import get_file_docs_for_qualifier
from app.services.agents.ai_qualifier import qualify_documents

router = APIRouter(prefix="/debug/qualifier", tags=["DEBUG – Qualifier"])


class QualifierDebugRequest(BaseModel):
    case_id: str
    as_pdf: bool = False


@router.post("/", summary="Полный отчёт pipeline квалификатора")
async def debug_qualifier(
    req: QualifierDebugRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:

    # ---------------------------------------------------------
    # 1) Retrieval (как боевой qualifier)
    # ---------------------------------------------------------
    docs = get_file_docs_for_qualifier(
        db=db,
        case_id=req.case_id
    )

    if not docs:
        return {
            "case_id": req.case_id,
            "error": "no_documents",
            "retrieval_docs": 0,
            "message": "Retrieval вернул 0 документов.",
        }

    retrieval_stats = {
        "docs_total": len(docs),
        "unique_files": len(set([d["file_id"] for d in docs])),
        "pages": list(set([d["page"] for d in docs])),
    }

    # ---------------------------------------------------------
    # 2) Запуск квалификатора (он сам внутри использует Tokenizer + FactGraph + Router)
    # ---------------------------------------------------------
    result = qualify_documents(
        case_id=req.case_id,
        docs=docs,
        city="г. Шымкент",
        investigator_fio="DEBUG",
        investigator_line="Отладочный режим",
        date_str=None,
    )


    # ---------------------------------------------------------
    # 3) Разворачиваем Debug-информацию (если модель вернула)
    # ---------------------------------------------------------
    debug_info = result.get("debug_info", {})

    return {
        "case_id": req.case_id,

        "retrieval": {
            "stats": retrieval_stats,
            "sample_chunks": docs[:5],  # первые 5 чанков для проверки
        },

        "qualifier_pipeline": {
            "tokens_extracted": debug_info.get("tokens_extracted"),
            "token_samples": debug_info.get("token_samples"),
            "factgraph_before": debug_info.get("factgraph_before"),
            "factgraph_after": debug_info.get("factgraph_after"),
            "facts_sent_to_llm": debug_info.get("facts_sent_to_llm"),
            "router_selected": debug_info.get("router_selected"),
            "crime_scores": debug_info.get("crime_scores"),
            "ustanovil": debug_info.get("ustanovil"),
            "postanovil": debug_info.get("postanovil"),
        },

        "final_output": result
    }
