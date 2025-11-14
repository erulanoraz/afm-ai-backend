# app/api/v1/schema.py
from fastapi import APIRouter
from app.db.models import Base
from app.utils.config import settings
from sqlalchemy import create_engine

router = APIRouter(prefix="/schema", tags=["Schema"])

engine = create_engine(settings.DB_URL, echo=False, future=True)


@router.post("/init")
def schema_init():
    """Создаёт все таблицы в БД (если их нет)."""
    try:
        Base.metadata.create_all(bind=engine)
        return {"status": "ok", "message": "✅ Таблицы успешно созданы."}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.post("/drop-report")
def schema_drop_report():
    """Удаляет все таблицы (осторожно — стирает данные)."""
    try:
        Base.metadata.drop_all(bind=engine)
        return {"status": "ok", "message": "⚠️ Все таблицы удалены."}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
