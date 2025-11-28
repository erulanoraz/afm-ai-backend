@echo off
title AFM Legal AI - Celery Workers

echo ==========================================
echo   STARTING ALL CELERY WORKERS
echo ==========================================

REM Запуск ingest worker 1
start "ingest1" cmd /k ^
celery -A app.worker.celery_app worker -Q default -n ingest1 --loglevel=INFO -P solo

REM Запуск ingest worker 2
start "ingest2" cmd /k ^
celery -A app.worker.celery_app worker -Q default -n ingest2 --loglevel=INFO -P solo

REM Запуск embeddings worker
start "embeddings" cmd /k ^
celery -A app.worker.celery_app worker -Q embeddings -n embed1 --loglevel=INFO -P solo

REM Запуск vectors worker
start "vectors" cmd /k ^
celery -A app.worker.celery_app worker -Q vectors -n vectors1 --loglevel=INFO -P solo

echo ==========================================
echo   ALL CELERY WORKERS STARTED
echo ==========================================

pause
