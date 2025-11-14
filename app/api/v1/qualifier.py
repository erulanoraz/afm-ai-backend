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
from app.services.agents.ai_qualifier import qualify_documents, LLMUnavailableError, validate_facts_completeness
from app.services.export.pdf_generator import generate_postanovlenie_pdf

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["AI Qualifier"])


# -----------------------------
# Модель запроса (минимальная)
# -----------------------------
class QualifyRequest(BaseModel):
    """Минимальная модель запроса для ИИ-квалификатора"""
    case_id: str = Field(..., description="Идентификатор дела", min_length=1)
    as_pdf: bool = Field(default=True, description="Вернуть результат в формате PDF (True/False)")

    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "255500121000018",
                "as_pdf": True
            }
        }


# -----------------------------
# Основной endpoint
# -----------------------------
@router.post(
    "/qualify",
    summary="Формирует постановление о квалификации деяния подозреваемого",
    responses={
        200: {
            "description": "Постановление успешно сформировано",
            "content": {"application/pdf": {}, "text/plain": {}}
        },
        404: {"description": "Документы для дела не найдены"},
        500: {"description": "Ошибка при формировании постановления"}
    }
)
def qualify_final_document(
    request: QualifyRequest = Body(...),
    db: Session = Depends(get_db),
):
    """
    🔹 Генерирует юридическое постановление о квалификации деяния подозреваемого.

    Процесс:
    1. Загрузка документов дела из БД
    2. Извлечение фактов, лиц, дат, сумм через regex и LLM
    3. Проверка полноты по ст. 204 УПК РК
    4. Формирование раздела «УСТАНОВИЛ»
    5. Генерация финального постановления
    6. Верификация результата
    7. Возврат PDF или текста
    """
    start_time = datetime.now()
    logger.info(f"Начало квалификации дела {request.case_id}")

    try:
        # 1️⃣ Устанавливаем текущую дату
        date_str = datetime.now().strftime("%d.%m.%Y")

        # 2️⃣ Загружаем документы
        logger.info(f"Загрузка документов для дела {request.case_id}")
        try:
            docs = get_file_docs_for_qualifier(db, case_id=request.case_id)
        except Exception as e:
            logger.error(f"Ошибка при загрузке документов: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка загрузки документов: {str(e)}")
        # 🧩 Проверяем полноту данных
        validate_facts_completeness(docs)


        if not docs:
            logger.warning(f"Документы для дела {request.case_id} не найдены")
            raise HTTPException(
                status_code=404,
                detail=f"Документы для дела {request.case_id} не найдены. "
                       f"Убедитесь, что файлы загружены и содержат текстовые данные."
            )

        logger.info(f"Загружено {len(docs)} документов")

        # 3️⃣ Запускаем AI-квалификатор
        logger.info("Запуск AI-квалификатора")
        try:
            result = qualify_documents(
                case_id=request.case_id,
                docs=docs,
                city="г. Павлодар",
                date_str=date_str,
                investigator_line="Следователь по особо важным делам",
                investigator_fio="",
            )
        except LLMUnavailableError as e:
            logger.error(f"LLM недоступен: {e}")
            raise HTTPException(status_code=503, detail=f"Сервис анализа временно недоступен: {str(e)}")
        except Exception as e:
            logger.error(f"Ошибка квалификатора: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Ошибка при анализе документов: {str(e)}")

        # 4️⃣ Проверка результата
        postanovlenie_body = result.get("final_postanovlenie", "").strip()
        if not postanovlenie_body or "[ОШИБКА:" in postanovlenie_body:
            error_msg = result.get("warnings", ["Неизвестная ошибка"])[0]
            logger.error(f"Квалификация не удалась: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Не удалось сформировать постановление: {error_msg}")

        # 5️⃣ Финальный текст
        final_text = _build_final_document(
            case_id=request.case_id,
            date_str=date_str,
            postanovlenie_body=postanovlenie_body,
            result=result
        )

        # 6️⃣ Возврат PDF или текста
        if request.as_pdf:
            try:
                pdf_bytes = generate_postanovlenie_pdf(final_text)
                logger.info(f"PDF сгенерирован ({len(pdf_bytes)} байт)")
                return StreamingResponse(
                    io.BytesIO(pdf_bytes),
                    media_type="application/pdf",
                    headers={
                        "Content-Disposition": f"attachment; filename=postanovlenie_{request.case_id}.pdf"
                    },
                )
            except Exception as e:
                logger.error(f"Ошибка генерации PDF: {e}")
                raise HTTPException(status_code=500, detail=f"Ошибка при создании PDF: {str(e)}")

        return PlainTextResponse(final_text, media_type="text/plain; charset=utf-8")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


# -----------------------------
# Проверка статуса дела
# -----------------------------
@router.get(
    "/qualify/status/{case_id}",
    summary="Проверяет наличие документов для квалификации дела",
    response_model=dict
)
def check_qualification_status(
    case_id: str,
    db: Session = Depends(get_db)
):
    """Проверяет, есть ли достаточно документов для анализа"""
    try:
        docs = get_file_docs_for_qualifier(db, case_id=case_id)
        return {
            "case_id": case_id,
            "ready": len(docs) > 0,
            "documents_count": len(docs),
            "message": (
                f"Дело готово к квалификации ({len(docs)} документов)"
                if docs
                else "Документы для дела не найдены"
            )
        }
    except Exception as e:
        logger.error(f"Ошибка проверки статуса: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при проверке статуса: {str(e)}")


# -----------------------------
# Формирование финального документа
# -----------------------------
def _build_final_document(
    case_id: str,
    date_str: str,
    postanovlenie_body: str,
    result: dict
) -> str:
    """Формирует финальный документ с метаданными"""

    # 🔹 Русская дата
    try:
        dt = datetime.strptime(date_str, "%d.%m.%Y")
        months = [
            "января", "февраля", "марта", "апреля", "мая", "июня",
            "июля", "августа", "сентября", "октября", "ноября", "декабря"
        ]
        rus_date = f"{dt.day} {months[dt.month - 1]} {dt.year} года"
    except Exception:
        rus_date = date_str

    # 🔹 Для подписи
    investigator_line = result.get("investigator_line", "Следователь")
    investigator_fio = result.get("investigator_fio", "")

    # 🔹 Верификация + предупреждения
    warnings_section = ""
    if result.get("warnings"):
        warnings_section = (
            "\n\n⚠️ ПРЕДУПРЕЖДЕНИЯ:\n" +
            "\n".join(f"• {w}" for w in result["warnings"])
        )

    verification_info = ""
    verification = result.get("verification", {})
    if not verification.get("overall_ok", True):
        verification_info = (
            "\n\n⚠️ ВНИМАНИЕ: Верификация выявила потенциальные несоответствия."
        )

    # 🔹 Финальный документ
    return f"""ПОСТАНОВЛЕНИЕ
о квалификации деяния подозреваемого

г. Павлодар, {rus_date}

{postanovlenie_body}

────────────────────────────────────────────────────────────

МЕТАДАННЫЕ АНАЛИЗА:
• ID генерации: {result.get('generation_id', 'N/A')}
• Дело: {case_id}
• Уверенность: {result.get('confidence', 0.0):.2%}
• Вердикт: {result.get('verdict', 'UNKNOWN')}
• Время: {result.get('timestamp', 'N/A')}
• Версия модели: {result.get('model_version', 'N/A')}
{warnings_section}
{verification_info}

Подпись:
Следователь: {investigator_line}
ФИО: {investigator_fio}
______________________
Дата: {rus_date}

Права подозреваемого, предусмотренные ст. 64 УПК РК:
- право знать, в чем он подозревается;
- право давать объяснения или отказаться от дачи объяснений;
- право пользоваться помощью защитника;
- право представлять доказательства;
- право заявлять ходатайства и отводы;
- право обжаловать действия и решения органа расследования.

Черновик сформирован автоматически с использованием AI.
Окончательное решение принимает следователь после проверки всех доказательств.""".strip()
