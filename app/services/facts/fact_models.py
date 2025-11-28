from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import uuid
import datetime


# ================================================================
# 📘 SourceRef — источник факта (файл, страница, позиция)
# ================================================================
class SourceRef(BaseModel):
    file_id: str
    page: int
    span: Optional[Tuple[int, int]] = None  # (start_char, end_char)


# ================================================================
# 📘 FactToken — атомарная единица (тот самый token)
# ================================================================
class FactToken(BaseModel):
    token_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str                    # amount / action / date / person / org / ...
    value: str                   # буквальная строка
    source: SourceRef            # откуда извлечено (file_id/page/span)

    class Config:
        extra = "forbid"


# ================================================================
# 📘 LegalFact — крупная структура, объединяющая токены
# ================================================================
class LegalFact(BaseModel):
    fact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # главный текст факта
    text: Optional[str] = None   # ← ЭТО КРИТИЧЕСКОЕ ПОЛЕ

    # старая копия — для совместимости
    span_text: Optional[str] = None

    # роль: money_transfer, victim_loss, suspect_action и т.д.
    role: Optional[str] = None

    # alias роли
    event_type: Optional[str] = None

    # токены внутри факта
    tokens: List[FactToken] = Field(default_factory=list)

    # источники (file_id/page)
    source_refs: List[SourceRef] = Field(default_factory=list)

    # индекс предложения
    sentence_index: Optional[int] = None

    # контекстное окно
    context_before: Optional[str] = None
    context_after: Optional[str] = None

    # подсказки для криминальной классификации
    article_hints: List[str] = Field(default_factory=list)

    # уверенность
    confidence: Optional[float] = None

    created_at: str = Field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat()
    )

    # -----------------------------------------------
    # Вспомогательные вычисляемые методы
    # -----------------------------------------------
    def token_ids(self) -> List[str]:
        return [t.token_id for t in self.tokens]

    def to_payload(self):
        """
        Удобно отправлять в LLM или Router
        """
        return {
            "fact_id": self.fact_id,
            "role": self.role,
            "tokens": [t.model_dump() for t in self.tokens],
            "source_refs": [s.model_dump() for s in self.source_refs],
            "span_text": self.span_text,
            "sentence_index": self.sentence_index,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "article_hints": self.article_hints,
            "confidence": self.confidence,
        }

    class Config:
        extra = "forbid"
