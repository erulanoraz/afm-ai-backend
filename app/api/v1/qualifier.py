# app/api/v1/qualifier.py
import io
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Body, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.services.retrieval import get_file_docs_for_qualifier
from app.services.agents.ai_qualifier import qualify_documents
from app.services.export.pdf_generator import generate_postanovlenie_pdf

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["AI Qualifier"])


# -----------------------------
# 📥 Модель запроса
# -----------------------------
class QualifyRequest(BaseModel):
    """Минимальная модель запроса для ИИ-квалификатора"""
    case_id: str = Field(..., min_length=1)
    as_pdf: bool = Field(default=False)   # <-- теперь False

    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "255500121000018",
                "as_pdf": False,         # <-- пример тоже False
            }
        }



# ============================================================
# 🔥 Основной endpoint квалификации
# ============================================================
@router.post(
    "/qualify",
    summary="Формирует постановление о квалификации деяния подозреваемого",
    responses={
        200: {
            "description": "Постановление успешно сформировано",
            "content": {
                "application/pdf": {},
                "text/plain": {},
            },
        },
        404: {"description": "Документы для дела не найдены"},
        500: {"description": "Ошибка при формировании постановления"},
    },
)
def qualify_final_document(
    request: QualifyRequest = Body(...),
    db: Session = Depends(get_db),
):
    start_time = datetime.now()
    logger.info(f"▶️ Начало квалификации дела {request.case_id}")

    try:
        # ------------------------------------------------------------
        # 1️⃣ Retrieval — забираем документы из БД
        #    ВАЖНО: здесь уже применены Chunker + OCR + Reranker
        # ------------------------------------------------------------
        logger.info(f"Загрузка документов для дела {request.case_id}")
        try:
            docs = get_file_docs_for_qualifier(db, case_id=request.case_id)
        except Exception as e:
            logger.error(f"Ошибка загрузки документов: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка загрузки документов: {str(e)}",
            )

        if not docs:
            raise HTTPException(
                status_code=404,
                detail=f"Документы для дела {request.case_id} не найдены.",
            )

        logger.info(f"Загружено документов для квалификации: {len(docs)}")

        # ------------------------------------------------------------
        # 2️⃣ Запуск AI Qualifier 4.4 (ChatGPT-style RAG)
        # ------------------------------------------------------------
        logger.info("🚀 Запуск AI Qualifier 4.4 (token-json)...")

        try:
            result = qualify_documents(
                case_id=request.case_id,
                docs=docs,
                city="г. Павлодар",
                date_str=datetime.now().strftime("%d.%m.%Y"),
                investigator_line="Следователь по особо важным делам",
                investigator_fio="",
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Ошибка в qualify_documents: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка анализа документов: {str(e)}",
            )

        # ------------------------------------------------------------
        # 3️⃣ Проверка результата
        # ------------------------------------------------------------
        postanovlenie_body = (result.get("final_postanovlenie") or "").strip()

        if not postanovlenie_body:
            logger.error("Квалификация не удалась: текст постановления пустой")
            raise HTTPException(
                status_code=500,
                detail="Квалификация не удалась: текст постановления пустой.",
            )

        # ------------------------------------------------------------
        # 4️⃣ Формирование финального текста
        # ------------------------------------------------------------
        final_text = _build_final_document(
            case_id=request.case_id,
            date_str=datetime.now().strftime("%d.%m.%Y"),
            postanovlenie_body=postanovlenie_body,
            result=result,
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"✔ Квалификация по делу {request.case_id} завершена за {duration:.1f} сек."
        )

        # ------------------------------------------------------------
        # 5️⃣ Возврат PDF или текста
        # ------------------------------------------------------------
        if request.as_pdf:
            try:
                pdf_bytes = generate_postanovlenie_pdf(final_text)
            except Exception as e:
                logger.error(f"Ошибка PDF генерации: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Ошибка при создании PDF: {str(e)}",
                )

            return StreamingResponse(
                io.BytesIO(pdf_bytes),
                media_type="application/pdf",
                headers={
                    "Content-Disposition": (
                        f"attachment; filename=postanovlenie_{request.case_id}.pdf"
                    )
                },
            )

        # если as_pdf = False → просто возвращаем текст
        return PlainTextResponse(
            final_text,
            media_type="text/plain; charset=utf-8",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка квалификации: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {str(e)}",
        )


# ============================================================
# 🔍 Проверка статуса дела
# ============================================================
@router.get(
    "/qualify/status/{case_id}",
    summary="Проверяет наличие документов для анализа",
    response_model=dict,
)
def check_qualification_status(
    case_id: str,
    db: Session = Depends(get_db),
):
    try:
        docs = get_file_docs_for_qualifier(db, case_id=case_id)
        return {
            "case_id": case_id,
            "ready": len(docs) > 0,
            "documents_count": len(docs),
        }
    except Exception as e:
        logger.error(f"Ошибка проверки статуса: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка проверки статуса: {str(e)}",
        )


# ============================================================
# 🧱 Формирование финального документа
# ============================================================
def _build_final_document(
    case_id: str,
    date_str: str,
    postanovlenie_body: str,
    result: dict,
) -> str:
    # красивая русская дата
    try:
        dt = datetime.strptime(date_str, "%d.%m.%Y")
        months = [
            "января",
            "февраля",
            "марта",
            "апреля",
            "мая",
            "июня",
            "июля",
            "августа",
            "сентября",
            "октября",
            "ноября",
            "декабря",
        ]
        rus_date = f"{dt.day} {months[dt.month - 1]} {dt.year} года"
    except Exception:
        rus_date = date_str

    generation_id = result.get("generation_id")
    model_version = result.get("model_version")
    timestamp = result.get("timestamp")

    investigator_line = result.get("investigator_line") or "Следователь"
    investigator_fio = result.get("investigator_fio") or ""

    # 🔹 НОВОЕ: берём текст «УСТАНОВИЛ» из результата Qualifier
    ustanovil_body = (result.get("established_text") or "").strip()

    # Если по какой-то причине пусто — не ломаемся, просто выводим только ПОСТАНОВИЛ
    if ustanovil_body:
        body_block = f"""УСТАНОВИЛ:
{ustanovil_body}

ПОСТАНОВИЛ:
{postanovlenie_body}"""
    else:
        body_block = f"""ПОСТАНОВИЛ:
{postanovlenie_body}"""

    return f"""ПОСТАНОВЛЕНИЕ
о квалификации деяния подозреваемого

г. Павлодар, {rus_date}

{body_block}

────────────────────────────────────────────────────────────

ID генерации: {generation_id}
Версия модели: {model_version}
Время генерации: {timestamp}

Следователь: {investigator_line}
ФИО: {investigator_fio}
______________________
Дата: {rus_date}

Черновик сформирован автоматически с использованием AI.
Окончательное решение принимает следователь.
""".strip()
