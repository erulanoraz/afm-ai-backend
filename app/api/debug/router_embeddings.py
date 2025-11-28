# app/api/debug/router_embeddings.py
from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np

from app.services.embeddings import embed_text

router = APIRouter(
    prefix="/debug/embeddings",
    tags=["DEBUG – Embeddings"],
)


class EmbeddingRequest(BaseModel):
    text: str


@router.post("/", summary="Проверка embed_text() через Weaviate")
async def debug_embeddings(req: EmbeddingRequest):
    """
    Debug-эндпоинт для проверки работы embed_text().

    Варианты поведения:
    - vector_dim > 0: Weaviate вернул вектор → семантический слой жив.
    - vector_dim = 0: либо индекс Chunk пуст, либо проблема с конфигом Weaviate.
    """
    vec = embed_text(req.text)

    if not vec:
        return {
            "vector_dim": 0,
            "vector_preview": [],
            "norm": 0.0,
            "error": "embedding_empty_or_index_empty",
            "message": (
                "embed_text вернул пустой вектор. "
                "Проверь, что чанки реально загружены в Weaviate "
                "через VectorClient.insert_chunk, и что включён text2vec-transformers."
            ),
        }

    arr = np.array(vec, dtype=np.float32)

    return {
        "vector_dim": len(vec),
        "vector_preview": vec[:10],
        "norm": float(np.linalg.norm(arr)),
    }
