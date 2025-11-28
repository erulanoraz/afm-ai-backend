# --------------------------------------------------------------
# DEBUG RAG Router 4.0 — Evidence Engine Style
# --------------------------------------------------------------
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import re

from app.services.embeddings import embed_batch


router = APIRouter(prefix="/debug/router", tags=["DEBUG – RAG Router"])


# ===============================
# MODELS
# ===============================
class FactItem(BaseModel):
    text: str
    role: str = "generic_fact"   # добавили поддержку ролей
    meta: Dict[str, Any] = {}


class RouterRequest(BaseModel):
    query: str
    facts: List[FactItem]
    top_k: int = 20


# ===============================
# HELPERS
# ===============================
def normalize(t: str) -> str:
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape or a.size == 0:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(a.dot(b) / denom)


# ===============================
# SCORING COMPONENTS
# ===============================
ROLE_WEIGHT = {
    "money_transfer": 0.50,
    "suspect_action": 0.45,
    "victim_loss": 0.40,
    "fraud_event": 0.35,
    "investment_event": 0.30,
    "generic_fact": 0.05,
    "background": -0.10,
}


SYNONYMS = {
    "transfer": [
        "перевел", "перевёл", "перевела", "перевели",
        "перевод", "перечислил", "отправил", "переводил",
    ],
    "fraud": [
        "обман", "ввел в заблуждение", "ввёл в заблуждение",
        "ввели в заблуждение", "заблуждение", "мошеннич",
        "обманул", "обманула", "обманули", "мфо-проект",
    ],
    "pyramid": [
        "пирамида", "финансовая схема", "инвест-проект"
    ]
}


def synonym_bonus(text: str) -> float:
    low = normalize(text)
    bonus = 0

    for syn in SYNONYMS["transfer"]:
        if syn in low:
            bonus += 0.20

    for syn in SYNONYMS["fraud"]:
        if syn in low:
            bonus += 0.30

    for syn in SYNONYMS["pyramid"]:
        if syn in low:
            bonus += 0.25

    return bonus


def length_penalty(text: str) -> float:
    L = len(text)
    if L < 50:
        return -0.05
    if L > 700:
        return -0.05
    return 0


# ===============================
# MAIN DEBUG ROUTER
# ===============================
@router.post("/", summary="RAG Router 4.0 — Evidence Engine Scoring")
async def debug_router(req: RouterRequest):

    if not req.facts:
        return {"error": "facts list empty"}

    texts = [req.query] + [f.text for f in req.facts]
    vectors = embed_batch(texts)

    if not vectors or any(v is None for v in vectors):
        return {"error": "embedding failed"}

    q_vec = np.array(vectors[0])
    fact_vecs = [np.array(v) for v in vectors[1:]]

    scored = []

    for idx, (fact, vec) in enumerate(zip(req.facts, fact_vecs)):
        tx = fact.text

        sim = cosine(q_vec, vec)
        syn = synonym_bonus(tx)
        lpen = length_penalty(tx)

        rweight = ROLE_WEIGHT.get(fact.role, 0)

        score = (
            sim * 0.7 +
            syn * 1.5 +
            rweight * 1.2 +
            lpen * 0.3
        )

        scored.append({
            "index": idx,
            "role": fact.role,
            "score": round(score, 6),
            "sim": round(sim, 6),
            "syn_bonus": round(syn, 4),
            "role_weight": rweight,
            "length_penalty": lpen,
            "text": tx[:180],
            "text_len": len(tx),
            "meta": fact.meta
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {
        "query": req.query,
        "results": scored[:req.top_k],
        "total": len(req.facts),
    }
