# app/api/debug/diagnostics.py

from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db import get_db
from app.services.retrieval import get_file_docs_for_qualifier
from app.services.facts.fact_tokenizer import FactTokenizer
from app.services.facts.fact_graph import FactGraph
from app.services.facts.fact_filter import FactFilter
from app.services.rag_router import RAGRouter

import logging

router = APIRouter(
    prefix="/debug/qualifier",
    tags=["DEBUG – Qualifier Diagnostics"]
)

logger = logging.getLogger("DIAGNOSTICS")


class DiagnosticsRequest(BaseModel):
    case_id: str



@router.post("/diagnostics")
def run_full_diagnostics(
    request: DiagnosticsRequest = Body(...),
    db: Session = Depends(get_db),
):
    case_id = request.case_id

    # --------------------------------------------------------
    # 1) RETRIEVAL
    # --------------------------------------------------------
    docs = get_file_docs_for_qualifier(db, case_id=case_id)

    retrieval_preview = []
    for d in docs[:10]:
        retrieval_preview.append({
            "file_id": d.get("file_id"),
            "page": d.get("page"),
            "text_preview": (d.get("text") or "")[:300]
        })

    # --------------------------------------------------------
    # 2) TOKENIZER
    # --------------------------------------------------------
    tokenizer = FactTokenizer()
    tokenized = tokenizer.tokenize(docs)

    tokenizer_preview = []
    for f in tokenized[:10]:
        tokenizer_preview.append({
            "text": f.text,
            "tokens": [(t.type, t.value) for t in f.tokens],
            "role": f.role,
            "confidence": f.confidence
        })

    # --------------------------------------------------------
    # 3) FACT GRAPH
    # --------------------------------------------------------
    graph = FactGraph()
    merged = graph.build(tokenized)

    graph_preview = []
    for f in merged[:10]:
        graph_preview.append({
            "text": f.text,
            "tokens": [(t.type, t.value) for t in f.tokens],
            "role": f.role
        })

    # --------------------------------------------------------
    # 4) FACT FILTER
    # --------------------------------------------------------
    filt = FactFilter()
    filtered = filt.filter_for_qualifier(merged)

    filter_preview = []
    for f in filtered[:10]:
        filter_preview.append({
            "text": f.text,
            "tokens": [(t.type, t.value) for t in f.tokens],
            "role": f.role
        })

    # --------------------------------------------------------
    # 5) RAG ROUTER
    # --------------------------------------------------------
    router_r = RAGRouter()
    routed = router_r.route_for_qualifier(filtered)

    router_preview = []
    for f in routed[:10]:
        router_preview.append({
            "text": f.text,
            "tokens": [(t.type, t.value) for t in f.tokens],
            "role": f.role,
            "confidence": f.confidence
        })

    # --------------------------------------------------------
    # 6) Формируем итог
    # --------------------------------------------------------
    return {
        "case_id": case_id,
        "retrieval": {
            "docs_total": len(docs),
            "examples": retrieval_preview
        },
        "tokenizer": {
            "total_facts": len(tokenized),
            "examples": tokenizer_preview
        },
        "fact_graph": {
            "total_facts": len(merged),
            "examples": graph_preview
        },
        "fact_filter": {
            "total_facts": len(filtered),
            "examples": filter_preview
        },
        "router": {
            "total_facts": len(routed),
            "examples": router_preview
        }
    }
