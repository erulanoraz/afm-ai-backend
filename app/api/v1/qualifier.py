import io
import logging
import re
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


# ============================================================
# 📥 Модель запроса (case_id УДАЛЁН)
# ============================================================
class QualifyRequest(BaseModel):
    as_pdf: bool = Field(default=False)

    class Config:
        json_schema_extra = {
            "example": {
                "as_pdf": False
            }
        }


# ============================================================
# 🔍 Автоматическое извлечение case_id из текста документов
# ============================================================
CASE_ID_REGEX = r"(\d{15})"

def extract_case_id_from_docs(docs):
    """
    Просматривает ВСЕ чанки и ищет номер ЕРДР.
    Возвращает строку из 15 цифр или "".
    """
    for d in docs:
        text = d.get("text") or ""
        m = re.search(CASE_ID_REGEX, text)
        if m:
            return m.group(1)
    return ""


# ============================================================
# 🔥 ENDPOINT квалификации
# ============================================================
@router.post(
    "/qualify",
    summary="Формирует постановление о квалификации деяния подозреваемого"
)
def qualify_final_document(
    request: QualifyRequest = Body(...),
    db: Session = Depends(get_db),
):
    start_time = datetime.now()
    logger.info("▶️ Начало квалификации (GLOBAL MODE — без case_id фильтра)")

    try:
        # ------------------------------------------------------------
        # 1) Retrieval GLOBAL — читаем ВСЕ файлы
        # ------------------------------------------------------------
        try:
            docs = get_file_docs_for_qualifier(db)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка Retrieval: {str(e)}"
            )

        if not docs:
            raise HTTPException(
                status_code=404,
                detail="Документы не найдены."
            )

        logger.info(f"📄 Документов для квалификации: {len(docs)}")

        # ------------------------------------------------------------
        # 2) Авто-извлечение case_id из всех текстов
        # ------------------------------------------------------------
        resolved_case_id = extract_case_id_from_docs(docs)
        if resolved_case_id:
            logger.info(f"🔎 Авто case_id найден: {resolved_case_id}")
        else:
            logger.warning("⚠️ case_id не найден в документах")
            resolved_case_id = ""   # пустой, но постановление все равно создадим

        # ------------------------------------------------------------
        # 3) Запуск AI Qualifier
        # ------------------------------------------------------------
        try:
            result = qualify_documents(
                case_id=resolved_case_id,
                docs=docs,
                city="",
                date_str=datetime.now().strftime("%d.%m.%Y"),
                investigator_line="Следователь по особо важным делам",
                investigator_fio="",
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка AI Qualifier: {str(e)}"
            )

        post_body = (result.get("final_postanovlenie") or "").strip()
        if not post_body:
            raise HTTPException(
                status_code=500,
                detail="Квалификация не удалась: пустой текст."
            )

        # ------------------------------------------------------------
        # 4) Генерация финального текста
        # ------------------------------------------------------------
        final_text = _build_final_document(
            case_id=resolved_case_id,
            date_str=datetime.now().strftime("%d.%m.%Y"),
            postanovlenie_body=post_body,
            result=result,
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"✔ Квалификация завершена за {duration:.1f} сек.")

        # ------------------------------------------------------------
        # 5) PDF или текст
        # ------------------------------------------------------------
        if request.as_pdf:
            try:
                pdf_bytes = generate_postanovlenie_pdf(final_text)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Ошибка PDF генерации: {str(e)}"
                )

            return StreamingResponse(
                io.BytesIO(pdf_bytes),
                media_type="application/pdf",
                headers={
                    "Content-Disposition": (
                        f"attachment; filename=postanovlenie_{resolved_case_id or 'unknown'}.pdf"
                    )
                },
            )

        return PlainTextResponse(final_text, media_type="text/plain; charset=utf-8")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка: {str(e)}"
        )


# ============================================================
# 🧱 Формирование финального текста
# ============================================================
def _build_final_document(
    case_id: str,
    date_str: str,
    postanovlenie_body: str,
    result: dict,
) -> str:

    # русская дата
    try:
        dt = datetime.strptime(date_str, "%d.%m.%Y")
        months = [
            "января", "февраля", "марта", "апреля", "мая", "июня",
            "июля", "августа", "сентября", "октября", "ноября", "декабря"
        ]
        rus_date = f"{dt.day} {months[dt.month - 1]} {dt.year} года"
    except Exception:
        rus_date = date_str

    # город
    city = (result.get("city") or "").strip()
    if city and not city.lower().startswith("г."):
        city = f"г. {city}"

    # поля из AI
    generation_id = result.get("generation_id")
    model_version = result.get("model_version")
    timestamp = result.get("timestamp")

    investigator_line = result.get("investigator_line") or "Следователь"
    investigator_fio = result.get("investigator_fio") or ""

    ustanovil_body = (result.get("established_text") or "").strip()

    # составление тела
    if ustanovil_body:
        body_block = f"""УСТАНОВИЛ:
{ustanovil_body}

ПОСТАНОВИЛ:
{postanovlenie_body}"""
    else:
        body_block = f"ПОСТАНОВИЛ:\n{postanovlenie_body}"

    # если case_id найден — пишем его заголовке
    case_line = f"по делу № {case_id}" if case_id else ""

    return f"""ПОСТАНОВЛЕНИЕ
о квалификации деяния подозреваемого {case_line}

{city}, {rus_date}

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
