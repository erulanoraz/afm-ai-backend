# app/api/debug/router_tokenizer.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any, List

from app.utils.sentence_splitter import split_into_sentences

router = APIRouter(
    prefix="/debug/tokenizer",
    tags=["DEBUG – Tokenizer"]
)


class TokenizerRequest(BaseModel):
    text: str


class TokenizerResponse(BaseModel):
    sentences_count: int
    sentences: List[Dict[str, Any]]


@router.post("/", summary="Debug: Sentence Splitter v14.0", response_model=TokenizerResponse)
async def debug_tokenizer(req: TokenizerRequest) -> TokenizerResponse:
    """
    Debug интерфейс для Sentence Splitter v14.0 (ChatGPT Evidence Engine style)
    """
    try:
        sentences: List[str] = split_into_sentences(req.text)

        return TokenizerResponse(
            sentences_count=len(sentences),
            sentences=[
                {
                    "index": idx,
                    "length": len(sentence),
                    "text": sentence
                }
                for idx, sentence in enumerate(sentences, start=1)
            ]
        )

    except (TypeError, ValueError) as e:
        return TokenizerResponse(
            sentences_count=0,
            sentences=[{"error": str(e)}]
        )
