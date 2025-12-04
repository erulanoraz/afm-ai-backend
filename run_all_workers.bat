@echo off
title AFM Legal AI - Celery Workers

echo ==========================================
echo   STARTING CELERY WORKERS
echo ==========================================

REM ------------------------------------------
REM WORKER 1 — INGEST (upload, OCR start, chunker)
REM ------------------------------------------
start "ingest" cmd /k ^
celery -A app.worker.celery_app worker -Q ingest -n worker_ingest --loglevel=INFO -P solo

REM ------------------------------------------
REM WORKER 2 — OCR (tesseract, clean text)
REM ------------------------------------------
start "ocr" cmd /k ^
celery -A app.worker.celery_app worker -Q ocr -n worker_ocr --loglevel=INFO -P solo

REM ------------------------------------------
REM WORKER 3 — VECTORS (Weaviate embeddings)
REM ------------------------------------------
start "vectors" cmd /k ^
celery -A app.worker.celery_app worker -Q vectors -n worker_vectors --loglevel=INFO -P solo

echo ==========================================
echo   CELERY WORKERS STARTED SUCCESSFULLY
echo   - ingest queue
echo   - ocr queue
echo   - vectors queue
echo ==========================================

pause
