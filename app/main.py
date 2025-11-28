# app/main.py
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.db.models import Base
from app.utils.config import settings
from app.api.v1 import upload, qualifier, schema


from app.api.debug.router_ingest import router as debug_ingest_router
from app.api.debug.router_ocr import router as debug_ocr_router
from app.api.debug.router_chunker import router as debug_chunker_router
from app.api.debug.router_embeddings import router as debug_embeddings_router
from app.api.debug.router_retrieval import router as debug_retrieval_router
from app.api.debug.router_reranker import router as debug_reranker_router
from app.api.debug.router_tokenizer import router as debug_tokenizer_router
from app.api.debug.router_facts import router as debug_facts_router
from app.api.debug.router_router import router as debug_router_router
from app.api.debug.router_qualifier import router as debug_qualifier_router
from app.api.debug.diagnostics import router as diagnostics_router



# ================================================================
# Настройка логирования
# ================================================================
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("afm_legal.log", encoding="utf-8")
    ]
)

logger = logging.getLogger("AFM_Legal")

# ==========================================
# 🔇 Полное отключение pdfminer логов
# ==========================================
import logging

PDFMINER_MODULES = [
    "pdfminer",
    "pdfminer.psparser",
    "pdfminer.pdfdocument",
    "pdfminer.pdftypes",
    "pdfminer.pdfinterp",
    "pdfminer.pdfpage",
    "pdfminer.cmapdb",
    "pdfminer.layout",
    "pdfminer.converter",
    "pdfminer.image",
]

for module in PDFMINER_MODULES:
    log = logging.getLogger(module)
    log.setLevel(logging.ERROR)
    log.propagate = False


# ================================================================
# Настройка базы данных
# ================================================================
try:
    engine = create_engine(
        settings.DB_URL,
        echo=settings.DEBUG,
        future=True,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Подключение к БД успешно настроено.")
except Exception as e:
    logger.error(f"Ошибка подключения к БД: {e}")
    raise


# ================================================================
# Lifecycle событий (startup/shutdown)
# ================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Жизненный цикл приложения AFM Legal AI"""
    logger.info("Запуск AFM Legal AI...")

    try:
        # Проверка и создание таблиц
        logger.info("Проверка структуры БД...")
        Base.metadata.create_all(bind=engine)
        logger.info("Таблицы и схемы готовы.")
    except Exception as e:
        logger.error(f"Ошибка при инициализации базы данных: {e}")
        raise

    # Проверка доступности LLM API (по желанию)
    try:
        import requests
        resp = requests.get(settings.LLM_API_URL.replace("/v1/chat/completions", "/health"), timeout=5)
        if resp.status_code == 200:
            logger.info("LLM API доступен.")
        else:
            logger.warning(f"LLM API ответил со статусом {resp.status_code}.")
    except Exception as e:
        logger.warning(f"Не удалось проверить LLM API: {e}")

    yield

    logger.info("Остановка AFM Legal AI...")
    try:
        engine.dispose()
        logger.info("Соединения с БД закрыты.")
    except Exception as e:
        logger.error(f"Ошибка при остановке: {e}")


# ================================================================
# Создание FastAPI приложения
# ================================================================
app = FastAPI(
    title=settings.APP_NAME or "AFM Legal AI",
    version=settings.APP_VERSION or "1.0",
    description="""
Система автоматической юридической квалификации дел для АФМ РК.

Возможности:
- Загрузка и анализ документов (PDF, DOCX, TXT)
- Извлечение фактов, лиц, дат и сумм
- Проверка полноты по ст. 204 УПК РК
- Автоматическая генерация постановлений
- Проверка достоверности (anti-hallucination)
- Экспорт постановления в PDF
""",
    lifespan=lifespan,
    debug=settings.DEBUG,
)

# ================================================================
# Middleware
# ================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене — только доверенные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# Роутеры
# ================================================================
app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
app.include_router(qualifier.router, tags=["AI Qualifier"])  # prefix внутри router
app.include_router(schema.router, prefix="/api/v1", tags=["Schema"])


# включение debug API
if settings.DEBUG:
    app.include_router(debug_ingest_router)
    app.include_router(debug_ocr_router)
    app.include_router(debug_chunker_router)
    app.include_router(debug_embeddings_router)
    app.include_router(debug_retrieval_router)
    app.include_router(debug_reranker_router)
    app.include_router(debug_tokenizer_router)
    app.include_router(debug_facts_router)
    app.include_router(debug_router_router)
    app.include_router(debug_qualifier_router)
    app.include_router(diagnostics_router)


# ================================================================
# Системные endpoints
# ================================================================
@app.get("/", tags=["System"])
def root():
    """Базовая информация о системе"""
    return {
        "message": "AFM Legal AI backend запущен успешно",
        "version": app.version,
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "upload": "/api/v1/upload",
            "qualify": "/api/v1/qualify",
            "schema": "/api/v1/schema"
        }
    }


@app.get("/health", tags=["System"])
def health_check():
    """Проверка состояния приложения и базы данных"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as e:
        logger.error(f"Ошибка подключения к БД: {e}")
        db_status = f"error: {str(e)}"

    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "database": db_status,
        "version": app.version,
    }


@app.get("/config", tags=["System"])
def get_config():
    """Возвращает конфигурацию приложения (без секретов)"""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "debug": settings.DEBUG,
        "llm_model": settings.LLM_MODEL,
        "llm_temperature": settings.LLM_TEMPERATURE,
        "llm_timeout": settings.LLM_TIMEOUT,
        "log_level": settings.LOG_LEVEL,
        "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
        "allowed_file_types": settings.ALLOWED_FILE_TYPES,
    }


# ================================================================
# Глобальный обработчик ошибок
# ================================================================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Глобальная обработка необработанных ошибок"""
    logger.error(f"Необработанная ошибка: {exc}", exc_info=True)
    return {
        "error": "internal_server_error",
        "message": "Внутренняя ошибка сервера",
        "detail": str(exc) if settings.DEBUG else "Обратитесь к администратору"
    }


# ================================================================
# Точка входа
# ================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
