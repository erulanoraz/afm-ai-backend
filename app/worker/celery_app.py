# app/worker/celery_app.py

import os
from pathlib import Path
from dotenv import load_dotenv

# ============================================================
# 1) ГРУЗИМ .env ДО импорта settings !!!
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]  # папка backend/
ENV_PATH = ROOT_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    print("⚠️ WARNING: .env файл не найден! Celery запустится без конфигурации.")

# ============================================================
# 2) Теперь можно импортировать settings
# ============================================================

from celery import Celery
from app.utils.config import settings

# ============================================================
# 3) Создаём Celery с правильными переменными окружения
# ============================================================

celery_app = Celery(
    "afm_legal_ai",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_routes={
        "app.worker.embedding_tasks.*": {"queue": "embeddings"},
        "app.tasks.vector_tasks.*": {"queue": "vectors"},
    },
    task_default_queue="default",
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    worker_concurrency=4,
    task_acks_late=True,
    broker_connection_retry_on_startup=True,
)

# ============================================================
# 4) Обязательный импорт тасков
# ============================================================
import app.tasks.ingest   # noqa
import app.worker.embedding_tasks  # noqa
import app.tasks.vector_tasks  # noqa

celery_app.autodiscover_tasks(["app.tasks", "app.worker"])
