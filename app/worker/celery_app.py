# app/worker/celery_app.py

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    print("⚠️ WARNING: .env файл не найден!")

from celery import Celery
from app.utils.config import settings

celery_app = Celery(
    "afm_legal_ai",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_routes={
        "app.tasks.vector_tasks.*": {"queue": "vectors"},
        # ❌ УДАЛИЛИ embedding_tasks из route!
    },
    task_default_queue="default",
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    worker_concurrency=4,
    task_acks_late=True,
    broker_connection_retry_on_startup=True,
)

# Импортируем ТОЛЬКО vector_tasks
import app.tasks.vector_tasks  # noqa

celery_app.autodiscover_tasks(["app.tasks"])
